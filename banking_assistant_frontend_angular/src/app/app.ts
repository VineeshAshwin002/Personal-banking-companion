import {
  AfterViewChecked,
  Component,
  ElementRef,
  signal,
  ViewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

interface ServerPayload {
  text?: string;
  answer?: string;
  message?: string;
}

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements AfterViewChecked {
  private readonly apiUrl = 'http://127.0.0.1:8000/chat';
  private shouldScrollMessages = false;

  @ViewChild('messagesContainer')
  private messagesContainer?: ElementRef<HTMLElement>;

  @ViewChild('messageInput')
  private messageInput?: ElementRef<HTMLTextAreaElement>;

  @ViewChild('customerIdInput')
  private customerIdInput?: ElementRef<HTMLInputElement>;

  protected customerId = '';
  protected messageInputValue = '';
  protected readonly isBusy = signal(false);
  protected readonly customerIdRequired = signal(false);
  protected readonly connectionStatus = signal('Ready to connect');
  protected readonly messages = signal<ChatMessage[]>([
    {
      role: 'assistant',
      text: 'Hello! How can I help you today? I can help with account details, credit-card information, personal information, security, and transactions.',
    },
  ]);

  protected readonly suggestions = [
    { label: 'Account details', question: 'Show my account details' },
    {
      label: 'Credit card outstanding',
      question: 'Show my credit card outstanding amount',
    },
    {
      label: 'Personal information',
      question: 'Show my registered email address',
    },
    {
      label: 'Account security',
      question: 'What security features are enabled on my account?',
    },
  ];

  ngAfterViewChecked(): void {
    if (this.shouldScrollMessages && this.messagesContainer) {
      const element = this.messagesContainer.nativeElement;
      element.scrollTop = element.scrollHeight;
      this.shouldScrollMessages = false;
    }
  }

  protected useSuggestion(question: string): void {
    this.messageInputValue = question;
    this.focusMessageInput();
    this.resizeMessageInput();
  }

  protected onMessageInput(): void {
    this.resizeMessageInput();
  }

  protected onCustomerIdInput(): void {
    if (this.customerId.trim()) {
      this.customerIdRequired.set(false);
    }
  }

  protected onMessageKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.submitMessage();
    }
  }

  protected submitMessage(): void {
    const message = this.messageInputValue.trim();
    if (!message || this.isBusy()) {
      return;
    }

    if (!this.customerId.trim()) {
      this.customerIdRequired.set(true);
      this.connectionStatus.set('Customer ID required');
      this.customerIdInput?.nativeElement.focus();
      return;
    }

    this.messageInputValue = '';
    this.resizeMessageInput();
    void this.sendMessage(message);
  }

  private async sendMessage(message: string): Promise<void> {
    this.messages.update((messages) => [
      ...messages,
      { role: 'user', text: message },
    ]);
    const answer: ChatMessage = { role: 'assistant', text: '' };
    this.messages.update((messages) => [...messages, answer]);
    this.requestScroll();
    this.setBusy(true);
    this.connectionStatus.set('Connecting...');

    try {
      const response = await fetch(this.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({
          message,
          customer_id: this.customerId.trim(),
        }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(
          `API returned ${response.status}${detail ? `: ${detail}` : ''}`,
        );
      }
      if (!response.body) {
        throw new Error('This browser does not support response streaming.');
      }

      this.connectionStatus.set('Connected');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullAnswer = '';
      let completedAnswer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split(/\r?\n\r?\n/);
        buffer = events.pop() ?? '';

        for (const rawEvent of events) {
          const parsed = this.parseServerEvent(rawEvent);
          if (!parsed) {
            continue;
          }

          if (parsed.eventName === 'chunk') {
            fullAnswer += parsed.payload.text ?? '';
            answer.text = fullAnswer;
            this.replaceLastAssistantMessage(answer.text);
            this.requestScroll();
          } else if (parsed.eventName === 'complete') {
            completedAnswer = parsed.payload.answer ?? '';
          } else if (parsed.eventName === 'error') {
            throw new Error(
              parsed.payload.message ??
                'The server could not process this request.',
            );
          }
        }
      }

      if (!fullAnswer && completedAnswer) {
        answer.text = completedAnswer;
        this.replaceLastAssistantMessage(answer.text);
      }
      if (!answer.text.trim()) {
        answer.text = 'The server finished without returning an answer.';
        this.replaceLastAssistantMessage(answer.text);
      }
    } catch (error: unknown) {
      this.connectionStatus.set('Connection error');
      const message =
        error instanceof Error ? error.message : 'Unknown connection error';
      answer.text = `${message}\n\nCheck that the FastAPI server is running at ${this.apiUrl} and that CORS allows this frontend origin.`;
      this.replaceLastAssistantMessage(answer.text);
    } finally {
      this.setBusy(false);
      this.focusMessageInput();
      this.requestScroll();
    }
  }

  private parseServerEvent(
    rawEvent: string,
  ): { eventName: string; payload: ServerPayload } | null {
    let eventName = 'message';
    const dataLines: string[] = [];

    for (const line of rawEvent.split(/\r?\n/)) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim());
      }
    }

    if (!dataLines.length) {
      return null;
    }

    const data = dataLines.join('\n');
    try {
      return { eventName, payload: JSON.parse(data) as ServerPayload };
    } catch {
      return { eventName, payload: { text: data } };
    }
  }

  private setBusy(busy: boolean): void {
    this.isBusy.set(busy);
  }

  private replaceLastAssistantMessage(text: string): void {
    this.messages.update((messages) => {
      const lastMessage = messages[messages.length - 1];
      if (!lastMessage || lastMessage.role !== 'assistant') {
        return messages;
      }

      return [...messages.slice(0, -1), { ...lastMessage, text }];
    });
  }

  private focusMessageInput(): void {
    this.messageInput?.nativeElement.focus();
  }

  private resizeMessageInput(): void {
    const input = this.messageInput?.nativeElement;
    if (!input) {
      return;
    }

    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
  }

  private requestScroll(): void {
    this.shouldScrollMessages = true;
  }
}
