from langgraph.graph import StateGraph , START , END
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage , AIMessage , BaseMessage , SystemMessage
# from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode, InjectedState
from typing import Optional
from typing_extensions import Annotated , List , TypedDict , Any
from dotenv import load_dotenv
import os
from langchain_core.tools import InjectedToolCallId
from langgraph.graph import add_messages
import json
import re
from datetime import datetime
from pathlib import Path
from langgraph.types import Send
from pydantic import Field , BaseModel

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://vineesh-ai-uae-resource.services.ai.azure.com/openai/v1",
)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT",
    "gpt-4.1-mini",
)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "text-embedding-3-small",
)
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_ENDPOINT",
    "https://vineesh-ai-uae-resource.services.ai.azure.com",
)

if not AZURE_OPENAI_API_KEY:
    raise RuntimeError(
        "AZURE_OPENAI_API_KEY is not set. "
        "Set it as an environment variable before running the application."
    )


print('Hello World...!!!')

# User Request -> Hide Username and Card Details -> GuardRail -> Agents -> Tools -> LLM -> Output

# Agents -> Account Details , Credit Card Details , Personal Information , Security Agents , AmountTransactionAgent

# [account_agent , credit_card_agent , personal_info_agent , security_agent , transaction_agent]

class AgentState(TypedDict):
    user_details : Any
    messages : Annotated[List[BaseMessage], add_messages]
    agents : List[str]
    user_request : str
    security: str 
    customer_id : str 

class SelectAgents(TypedDict):
    agents : List[str]
    user_request : str


class Agents(BaseModel):
    agents : List[str] = Field(description="agents required to fulfill the user's request.")

def call_embeddings():
    try:
        return AzureOpenAIEmbeddings(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            base_url=AZURE_OPENAI_EMBEDDING_ENDPOINT
        )

    except Exception as e:
        print(f"Error ",e)


def call_llm():
    try:
        return ChatOpenAI(
            model=AZURE_OPENAI_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            base_url=AZURE_OPENAI_ENDPOINT,
            temperature=0.5,
        )


    except Exception as e:
        print(f" Error ", e)


# Keep the data file location independent of the process working directory.
# This module lives in banking_assistant_backend/, while users_details.json is
# stored at the project root. Local deployments can keep the real data in
# users_details.json; fresh clones use the safe example data file.
DATA_DIRECTORY = Path(__file__).resolve().parent.parent
BANK_DATA_FILE = DATA_DIRECTORY / "users_details.json"
if not BANK_DATA_FILE.exists():
    BANK_DATA_FILE = DATA_DIRECTORY / "users_details.example.json"


def load_bank_data():
    with BANK_DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_bank_data(data):
    temp_file = BANK_DATA_FILE.with_name(f"{BANK_DATA_FILE.name}.tmp")

    with temp_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    os.replace(temp_file, BANK_DATA_FILE)

def orchestrator(state: Annotated[AgentState, InjectedState]) -> AgentState:
    try:

        print("User Request: ", state["user_request"])
        PROMPT = """

You are the Banking Multi-Agent Orchestrator.

Analyze the user's request and select the agent(s) required to completely fulfill it.

User Request:
{user_request}

AVAILABLE AGENTS:

ACCOUNT_AGENT
- Account balance / available balance
- Minimum balance
- Transaction limits
- Account status
- General account information

CREDIT_CARD_AGENT
- Credit card limit / available credit
- Credit card balance / outstanding
- Minimum due / payment due date
- Credit card transactions and information

PERSONAL_INFORMATION_AGENT
- View or update name, email, phone, address
- Customer profile information

SECURITY_AGENT
- Security checks
- Security settings or updates
- Account security operations

AMOUNT_TRANSACTION_AGENT
- Withdraw money
- Transfer / send funds
- Deposit money
- Any operation that moves money


ROUTING RULES:

1. Select only the agent(s) required to completely answer or perform the user's request.
2. If one agent is sufficient, return one agent.
3. If the request requires information or operations from multiple agents, return all required agents.
4. Do not select an agent just because its information is mentioned; select it only when its data or operation is required.
5. If operations are independent, they can be executed in parallel.
6. If one operation depends on another, preserve the required execution order.
7. Do not execute tools or answer the request. Only determine the required agent(s).
8. If the request cannot be confidently mapped to an agent, return NEEDS_CLARIFICATION.

EXAMPLES:

"What is my account balance?"
→ ["ACCOUNT_AGENT"]

"What is my credit card outstanding?"
→ ["CREDIT_CARD_AGENT"]

"Update my phone number."
→ ["PERSONAL_INFORMATION_AGENT"]

"Check my account security."
→ ["SECURITY_AGENT"]

"Transfer ₹10,000 to Arun."
→ ["AMOUNT_TRANSACTION_AGENT"]

"Check my balance and credit card outstanding."
→ ["ACCOUNT_AGENT", "CREDIT_CARD_AGENT"]

"Transfer ₹20,000 to Arun and check whether I have enough balance."
→ ["ACCOUNT_AGENT", "AMOUNT_TRANSACTION_AGENT"]

"Show my profile and check my security settings."
→ ["PERSONAL_INFORMATION_AGENT", "SECURITY_AGENT"]

FINAL OUTPUT:

Return ONLY valid JSON:

{{
  "agents": ["AGENT_NAME"]
}}

or:

{{
  "agents": ["AGENT_NAME_1", "AGENT_NAME_2"]
}}

If clarification is required:

{{
  "agents": ["NEEDS_CLARIFICATION"]
}}

"""


        llm = call_llm()

        structured_llm = llm.with_structured_output(Agents)

        result = structured_llm.invoke(
            [SystemMessage(content=PROMPT.format(user_request=state["user_request"]))]
        )
        
        # result = llm.with_structured_output(
        #     [SystemMessage(content=PROMPT.format(user_request=state["user_request"]))]
        # )

        print("Result from Orchestrator: ", result.agents)


        update_text_file(state['messages'])
        
        # state['agents'] = result.agents
        # state_1["messages"].append(SystemMessage(content=PROMPT.format(user_request=state["user_request"])))

        return {
            "agents":  result.agents ,
            'messages': [
                SystemMessage(content=PROMPT.format(user_request=state["user_request"])),
                # AIMessage(content=result)
                AIMessage(content=f"Selected Agents {' '.join(result.agents)}")
            ]
        }

    except Exception as e:
        print(f" Error {e}")
        raise

def orchestrator_2(state: AgentState) -> str:
    return "security_agent" if str(state.get("customer_id", "")).strip() else "end"


def secrity_agent(state: AgentState) -> dict:
    """Pre-routing security classifier; do not confuse with SECURITY_AGENT."""
    prompt_text = """Classify the user's request as exactly Malicious or Non-Malicious.
Malicious means clear intent to gain unauthorized access, steal credentials, bypass controls, deploy malware, or harm systems.
Defensive, educational, ordinary programming, and authorized banking requests are Non-Malicious.
Return only one exact label.
User request: {user_request}
""".format(user_request=state.get("user_request", ""))
    result = call_llm().invoke([SystemMessage(content=prompt_text)])
    classification = str(result.content).strip()
    if classification not in {"Malicious", "Non-Malicious"}:
        classification = "Non-Malicious"
    print("Security classification:", classification)
    return {"security": classification, "messages": [result]}


def orchestrator_1(state: AgentState) -> str:
    return END if str(state.get("security", "")).strip().lower() == "malicious" else "orchestrator"


def security_operations_agent(state: AgentState) -> dict:
    """Handle SECURITY_AGENT requests without inventing unavailable settings."""
    response = call_llm().invoke([
        SystemMessage(content=(
            "You are a banking security support assistant. Answer general security questions safely. "
            "The connected sample customer data does not provide security settings or security-operation tools, "
            "so do not claim to inspect or change settings. If the user asks to change or inspect a setting, "
            "explain that the operation is not available in this demo and suggest using the bank's official app/support."
        )),
        HumanMessage(content=state.get("user_request", "")),
    ])
    return {"messages": [response]}


def clarification_agent(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=(
        "I need a little more detail to route this request. Please specify whether you need account information, "
        "credit-card information/payment, personal-information changes, security help, or a money transfer."
    ))]}


def sanitize_customer_data(customer):
    """
    Remove sensitive customer information before sending data to the LLM.
    Keeps only information required for reasoning.
    """

    return {
        "account": {
            "account_type": customer["account"]["account_type"],
            "currency": customer["account"]["currency"],
            "balance": customer["account"]["balance"],
            "minimum_balance": customer["account"]["minimum_balance"],
            "available_balance": customer["account"]["available_balance"],
            "status": customer["account"]["status"],
        },

        "credit_cards": [
            {
                "card_type": card["card_type"],
                "card_name": card["card_name"],
                "masked_card_number": card["masked_card_number"],
                "credit_limit": card["credit_limit"],
                "available_credit": card["available_credit"],
                "outstanding_amount": card["outstanding_amount"],
                "minimum_due": card["minimum_due"],
                "statement_date": card["statement_date"],
                "payment_due_date": card["payment_due_date"],
                "status": card["status"],
                "reward_points": card["reward_points"],
            }
            for card in customer.get("credit_cards", [])
        ],

        "beneficiaries": [
            {
                "name": beneficiary["name"],
                "status": beneficiary["status"],
            }
            for beneficiary in customer.get("beneficiaries", [])
        ],

        "transaction_limits": customer.get("transaction_limits", {})
    }


# AMOUNT_TRANSACTION_AGENT , PERSONAL_INFORMATION_AGENT , CREDIT_CARD_AGENT , ACCOUNT_AGENT 


def amount_transaction_agent(state: Annotated[AgentState, InjectedState]) -> dict:
    """Handle a transfer using details extracted from the frontend request.

    No terminal input is used. The user request must include a transfer amount
    and a beneficiary name; otherwise the agent asks the frontend user for them.
    """
    try:
        customer_id = str(state.get("customer_id") or "").strip()
        if not customer_id:
            return {"messages": [AIMessage(content="Please provide your customer ID before requesting a transfer.")]}

        data = load_bank_data()
        customer = next(
            (item for item in data.get("customers", [])
             if item.get("customer_id") == customer_id),
            None,
        )
        if not customer:
            return {"messages": [AIMessage(content="I couldn't verify that customer ID. Please check it and try again.")]}

        class TransferRequest(BaseModel):
            transfer_amount: Optional[float] = Field(
                default=None,
                description="The amount the user explicitly requested to transfer. Use null if not provided."
            )
            beneficiary_name: Optional[str] = Field(
                default=None,
                description="The beneficiary name explicitly stated by the user. Use null if not provided."
            )

        extractor = call_llm().with_structured_output(TransferRequest)
        details = extractor.invoke([
            SystemMessage(content=(
                "Extract only the transfer amount and beneficiary name explicitly stated "
                "in the user's message. Never guess or invent either value. Use null "
                "for missing information."
            )),
            HumanMessage(content=state.get("user_request", "")),
        ])

        if details.transfer_amount is None or not details.beneficiary_name:
            return {"messages": [AIMessage(content=(
                "To make a transfer, please send the amount and beneficiary name together. "
                "For example: 'Transfer ₹1,000 to Arun.'"
            ))]}

        transfer_amount = float(details.transfer_amount)
        beneficiary_name = details.beneficiary_name.strip()

        account = customer.get("account", {})
        if not account:
            return {"messages": [AIMessage(content="No account information is available for this customer.")]}

        try:
            available_balance = float(account.get("available_balance", 0))
        except (TypeError, ValueError):
            return {"messages": [AIMessage(content="The available balance could not be verified. The transfer was not made.")]}

        if transfer_amount <= 0:
            return {"messages": [AIMessage(content="The transfer amount must be greater than zero. No transfer was made.")]}

        if transfer_amount > available_balance:
            return {"messages": [AIMessage(content=(
                f"Insufficient available balance. Your available balance is ₹{available_balance:.2f}. "
                "No transfer was made."
            ))]}

        beneficiaries = customer.get("beneficiaries", [])
        beneficiary = next(
            (item for item in beneficiaries
             if str(item.get("name", "")).casefold() == beneficiary_name.casefold()),
            None,
        )
        if not beneficiary:
            available_names = ", ".join(
                str(item.get("name", "")) for item in beneficiaries if item.get("name")
            )
            if available_names:
                return {"messages": [AIMessage(content=(
                    f"'{beneficiary_name}' is not in your saved beneficiaries. "
                    f"Available beneficiaries: {available_names}. No transfer was made."
                ))]}
            return {"messages": [AIMessage(content="No saved beneficiaries were found. Add a beneficiary before transferring money. No transfer was made.")]}

        previous_balance = available_balance
        updated_balance = previous_balance - transfer_amount
        account["available_balance"] = updated_balance
        account["balance"] = updated_balance + float(account.get("minimum_balance", 0) or 0)

        save_bank_data(data)

        receipt = (
            "Transfer completed successfully.\n"
            f"Beneficiary: {beneficiary.get('name')}\n"
            f"Amount: ₹{transfer_amount:.2f}\n"
            f"Previous available balance: ₹{previous_balance:.2f}\n"
            f"Updated available balance: ₹{updated_balance:.2f}\n"
            "Status: Successful"
        )
        state["transaction_amount"] = transfer_amount
        state["updated_balance"] = updated_balance
        state["beneficiary"] = beneficiary.get("name")
        state["transaction_receipt"] = receipt
        update_text_file([AIMessage(content=receipt)])
        return {"messages": [AIMessage(content=receipt)]}

    except Exception:
        return {"messages": [AIMessage(content="Unable to process the transfer. Please check the details and try again. No further action was taken.")]}

def run_tool_agent(state: AgentState, system_prompt: str, tools: list) -> dict:
    """Run one tool-using agent, executing only its own tool calls.

    Tool messages are kept in this agent's local conversation so parallel
    agents cannot accidentally execute one another's tool calls.
    """
    llm = call_llm()
    llm_with_tools = llm.bind_tools(tools)
    conversation = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.get("user_request", "")),
    ]
    tool_node = ToolNode(tools)

    for _ in range(5):
        response = llm_with_tools.invoke(conversation)
        conversation.append(response)
        if not getattr(response, "tool_calls", None):
            break

        tool_result = tool_node.invoke({**state, "messages": [response]})
        tool_messages = tool_result.get("messages", [])
        conversation.extend(tool_messages)
    else:
        conversation.append(AIMessage(content="I couldn't complete the request after several tool steps."))

    final_message = conversation[-1]
    if isinstance(final_message, AIMessage) and final_message.tool_calls:
        final_message = AIMessage(content="I couldn't complete the request because the tool-call limit was reached.")
        conversation.append(final_message)

    update_text_file(conversation)
    return {"messages": conversation}

def credit_card_agent(state: AgentState) -> dict:
    """Choose and execute the appropriate credit-card tool."""
    system_prompt = """You are the CREDIT_CARD_AGENT. Handle only credit-card requests.
Use `credit_card` for all read-only requests (details, limit, available credit, outstanding amount, minimum due, bill, status).
Use `outstanding_amount_pay` ONLY when the user explicitly asks to make/pay a credit-card payment.
Extract the exact payment amount from the user's message and pass it as `payment_amount`.
If the user did not state an amount, ask them to include it; do not call the payment tool.
Never use the payment tool for questions about due amounts. Do not invent values.
"""
    print("Reached Credit Card Agent....")
    return run_tool_agent(state, system_prompt, [credit_card, outstanding_amount_pay])


@tool
def credit_card(state: Annotated[AgentState, InjectedState]) -> str:
    """
    Read and return credit card details for a customer.
    Use this for credit card limit, available credit, status,
    card details and general credit card information.
    """

    try:

        print("Credit Card Tool Reached....")

        data = load_bank_data()

        customer = next(
            (
                customer
                for customer in data["customers"]
                if customer["customer_id"] == state['customer_id']
            ),
            None
        )

        if not customer:
            return f"Customer {state['customer_id']} was not found."

        card = customer.get("credit_cards")

        if not card:
            return "No credit card information is available for this customer."

        # # Don't expose sensitive card information unnecessarily
        # safe_card_data = {
        #     "card_number": (
        #         f"**** **** **** {card['card_number'][-4:]}"
        #         if card.get("card_number")
        #         else None
        #     ),
        #     "credit_limit": card.get("credit_limit"),
        #     "available_credit": card.get("available_credit"),
        #     "outstanding_amount": card.get("outstanding_amount"),
        #     "minimum_due": card.get("minimum_due"),
        #     "status": card.get("status")
        # }

        # Don't expose sensitive card information unnecessarily
        safe_card_data = []

        for card_data in card:
            safe_card_data.append({
                "card_number": (
                    f"**** **** **** {card_data['card_number'][-4:]}"
                    if card_data.get("card_number")
                    else None
                ),
                "credit_limit": card_data.get("credit_limit"),
                "available_credit": card_data.get("available_credit"),
                "outstanding_amount": card_data.get("outstanding_amount"),
                "minimum_due": card_data.get("minimum_due"),
                "status": card_data.get("status")
            })

        prompt = f"""
You are a banking credit card assistant.

Generate a clear and professional response for the customer.

Customer:
{customer.get("name")}

Credit Card Details:
{safe_card_data}

Rules:
- Answer only using the provided data.
- Do not invent any values.
- Do not expose the complete credit card number.
- Explain the information clearly.
- If the user asks for credit card details, provide the relevant details.
- Keep the response concise.

Generate only the response that should be shown to the customer.
"""

        llm = call_llm()

        response = llm.invoke(prompt)

        update_text_file(state['messages'])

        return response.content

    except Exception as e:
        return f"Unable to retrieve credit card details: {str(e)}"


@tool
def outstanding_amount_pay(
    state: Annotated[AgentState, InjectedState],
    payment_amount: float,
) -> str:
    """Pay a credit-card amount explicitly stated by the user.

    The payment amount must be supplied from the user's frontend request.
    Never ask for input in the terminal.
    """
    try:
        data = load_bank_data()
        customer_id = str(state.get("customer_id") or "").strip()
        customer = next(
            (item for item in data.get("customers", [])
             if item.get("customer_id") == customer_id),
            None,
        )
        if not customer:
            return "Customer could not be verified. No payment was made."

        cards = customer.get("credit_cards", [])
        if not cards:
            return "No credit card information is available for this customer."

        card = cards[0]
        outstanding_amount = float(card.get("outstanding_amount", 0) or 0)
        available_credit = float(card.get("available_credit", 0) or 0)
        payment_amount = float(payment_amount)

        if payment_amount <= 0:
            return "The payment amount must be greater than zero. No payment was made."
        if outstanding_amount <= 0:
            return "There is no outstanding amount to pay."
        if payment_amount > outstanding_amount:
            return (
                f"Payment amount cannot exceed the outstanding amount of "
                f"₹{outstanding_amount:.2f}. No payment was made."
            )

        updated_outstanding = outstanding_amount - payment_amount
        updated_available_credit = available_credit + payment_amount
        minimum_due = float(card.get("minimum_due", 0) or 0)
        updated_minimum_due = max(minimum_due - payment_amount, 0)

        card["outstanding_amount"] = updated_outstanding
        card["available_credit"] = updated_available_credit
        card["minimum_due"] = updated_minimum_due
        save_bank_data(data)

        masked_card = (
            f"**** **** **** {card['card_number'][-4:]}"
            if card.get("card_number") else "Not available"
        )
        return (
            "Credit card payment processed successfully.\n"
            f"Card: {masked_card}\n"
            f"Payment amount: ₹{payment_amount:.2f}\n"
            f"Previous outstanding: ₹{outstanding_amount:.2f}\n"
            f"Remaining outstanding: ₹{updated_outstanding:.2f}\n"
            f"Updated available credit: ₹{updated_available_credit:.2f}\n"
            f"Updated minimum due: ₹{updated_minimum_due:.2f}\n"
            "Status: Successful"
        )
    except (TypeError, ValueError):
        return "Please provide a valid numeric payment amount. No payment was made."
    except Exception:
        return "Unable to process the credit card payment. Please try again later."


@tool
def read_personal_details(state: Annotated[AgentState, InjectedState]) -> str:
    """Read the verified customer's personal details using customer_id from state.

    Do not ask for account number or confirmation in the terminal.
    """
    try:
        data = load_bank_data()
        customer_id = str(state.get("customer_id") or "").strip()
        if not customer_id:
            return "Please provide your customer ID to retrieve personal information."

        customer = next(
            (item for item in data.get("customers", [])
             if item.get("customer_id") == customer_id),
            None,
        )
        if not customer:
            return "Customer could not be verified. Personal information was not returned."

        account = customer.get("account", {})
        account_number = str(account.get("account_number", ""))
        masked_account = (
            f"****{account_number[-4:]}" if len(account_number) >= 4
            else "Not available"
        )
        personal_details = {
            "name": customer.get("name"),
            "email": customer.get("email"),
            "phone_number": customer.get("phone"),
            "account_number": masked_account,
            "account_type": account.get("account_type"),
            "beneficiaries": [
                {"name": b.get("name"), "status": b.get("status")}
                for b in customer.get("beneficiaries", [])
            ],
        }
        prompt = f"""
You are a banking personal-information assistant.
User request: {state.get("user_request", "")}
Verified customer details:
{personal_details}

Answer only the user's question using the provided information.
Do not invent information or reveal the full account number.
Keep the answer concise.
"""
        response = call_llm().invoke(prompt)
        return response.content
    except Exception:
        return "Unable to retrieve personal details right now."


@tool
def write_personal_details(
    state: Annotated[AgentState, InjectedState],
    field: str,
    value: str = "",
    beneficiary_name: str = "",
    beneficiary_account_number: str = "",
    bank_name: str = "",
    ifsc: str = "",
) -> str:
    """Update a personal field or add a beneficiary using values from the user's request.

    For name/email/phone updates, pass field and value.
    For beneficiary creation, set field='beneficiary' and provide beneficiary_name,
    beneficiary_account_number, bank_name, and ifsc.
    Never prompt in the terminal. If required details are missing, return a message
    asking the user to send the missing information through the frontend.
    """
    try:
        data = load_bank_data()
        customer_id = str(state.get("customer_id") or "").strip()
        if not customer_id:
            return "Please provide your customer ID before updating personal information."

        customer = next(
            (item for item in data.get("customers", [])
             if item.get("customer_id") == customer_id),
            None,
        )
        if not customer:
            return "Customer could not be verified. No changes were made."

        normalized_field = (field or "").strip().lower()
        aliases = {"phone": "phone_number", "mobile": "phone_number"}
        normalized_field = aliases.get(normalized_field, normalized_field)

        if normalized_field in {"name", "email", "phone_number"}:
            new_value = (value or "").strip()
            if not new_value:
                return f"Please provide the new {normalized_field.replace('_', ' ')} in your message. No changes were made."
            if normalized_field == "email" and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", new_value):
                return "Please provide a valid email address. No changes were made."
            if normalized_field == "phone_number" and len(re.sub(r"\D", "", new_value)) < 7:
                return "Please provide a valid phone number. No changes were made."

            key = "phone" if normalized_field == "phone_number" else normalized_field
            old_value = customer.get(key)
            customer[key] = new_value
            update_data = {
                "field": normalized_field,
                "old_value": old_value,
                "new_value": new_value,
                "status": "Updated",
            }

        elif normalized_field == "beneficiary":
            values = [
                (beneficiary_name or "").strip(),
                (beneficiary_account_number or "").strip(),
                (bank_name or "").strip(),
                (ifsc or "").strip(),
            ]
            if not all(values):
                return (
                    "To add a beneficiary, send the beneficiary name, account number, "
                    "bank name, and IFSC code together. No changes were made."
                )
            ifsc_value = (ifsc or "").strip().upper()
            beneficiaries = customer.setdefault("beneficiaries", [])
            existing_ids = []
            for beneficiary in beneficiaries:
                beneficiary_id = str(beneficiary.get("beneficiary_id", ""))
                match = re.fullmatch(r"BEN(\d+)", beneficiary_id)
                if match:
                    existing_ids.append(int(match.group(1)))
            beneficiary_id = f"BEN{max(existing_ids, default=2000) + 1}"
            new_beneficiary = {
                "beneficiary_id": beneficiary_id,
                "name": beneficiary_name.strip(),
                "account_number": beneficiary_account_number.strip(),
                "bank_name": bank_name.strip(),
                "ifsc": ifsc_value,
                "status": "Active",
            }
            beneficiaries.append(new_beneficiary)
            update_data = {
                "field": "beneficiary",
                "beneficiary": new_beneficiary,
                "status": "Added",
            }
        else:
            return "Unsupported field. You can update name, email, phone number, or add a beneficiary. No changes were made."

        save_bank_data(data)
        return (
            "Personal information updated successfully.\n"
            + json.dumps(update_data, ensure_ascii=False, indent=2)
        )
    except Exception:
        return "Unable to update personal information. Please try again later."

def personal_detail_agent(state: AgentState) -> dict:
    """Choose and execute the appropriate personal-information tool."""
    system_prompt = """You are the PERSONAL_INFORMATION_AGENT. Handle only customer profile and beneficiary requests.
Use `read_personal_details` to view personal details, name, email, phone, address, or beneficiaries.
Use `write_personal_details` only when the user explicitly asks to change personal information or add a beneficiary.
For name/email/phone updates, pass `field` and the new `value`.
For beneficiary creation, pass field='beneficiary' plus beneficiary_name, beneficiary_account_number, bank_name, and ifsc extracted from the user's message.
If required values are missing, ask the user to provide them through the chat and do not call the tool.
Do not answer from guesses; use the appropriate tool.
"""
    print("Reached Personal Detail Agent....")
    return run_tool_agent(state, system_prompt, [read_personal_details, write_personal_details])


def account_agent(state: AgentState) -> dict:
    """Choose and execute the appropriate account tool."""
    system_prompt = """You are the ACCOUNT_AGENT. Handle general bank-account requests only, never credit-card requests.
Use `read_account_details` for account information, account status/type, balance, account number, beneficiaries, and related read-only details.
Use `raise_issue_ticket` only when the user wants to report a problem or create a support ticket; pass the issue description from the user's message as `issue`.
If the user has not described the issue, ask them to describe it in chat rather than calling the tool.
Do not invent information or modify account data. Use the appropriate tool.
"""
    print("Reached Account Agent....")
    return run_tool_agent(state, system_prompt, [read_account_details, raise_issue_ticket])


@tool
def read_account_details(state: Annotated[AgentState , InjectedState]) -> AgentState:
    """
    Read all customer/account details except credit card information.
    """

    try:
        data = load_bank_data()

        customer = next(
            (
                customer
                for customer in data.get("customers", [])
                if customer.get("customer_id") == state["customer_id"]
            ),
            None
        )

        if not customer:
            return "Customer not found."

        # Remove credit card information
        account_details = {
            key: value
            for key, value in customer.items()
            if key not in {"credit_card", "credit_cards"}
        }

        prompt = f"""
You are a banking account assistant.

User Request : {state['user_request']}

The following are the verified customer account details:

{account_details}

Generate a clear and professional response to the customer.

Rules:
- Use only the information provided above.
- Do not invent any information.
- Do not mention or expose credit card information.
- Do not expose the complete sensitive information unnecessarily.
- If the user asked for a specific account detail, provide only the relevant information.
- Keep the response concise and easy to understand.

Generate only the customer-facing response.
"""

        #llm = call_llm()


        llm = call_llm()

        response = llm.invoke(prompt)

        print("Response from read_account_details:", response.content)
        update_text_file(state['messages'])

        return response.content

    except Exception as e:
        return f"Unable to retrieve account details: {str(e)}"



@tool
def raise_issue_ticket(
    state: Annotated[AgentState, InjectedState],
    issue: str,
) -> str:
    """Create a support ticket for the issue explicitly described by the user."""
    try:
        data = load_bank_data()
        customer_id = str(state.get("customer_id") or "").strip()
        if not customer_id:
            return "Please provide your customer ID before raising a support ticket."
        customer = next(
            (item for item in data.get("customers", [])
             if item.get("customer_id") == customer_id),
            None,
        )
        if not customer:
            return "Customer could not be verified. No ticket was created."
        issue = (issue or "").strip()
        if not issue:
            return "Please describe the issue in your message. No ticket was created."

        tickets = data.setdefault("issue_tickets", [])
        ticket_numbers = []
        for ticket in tickets:
            ticket_id = str(ticket.get("ticket_id", ""))
            match = re.fullmatch(r"TKT(\d+)", ticket_id)
            if match:
                ticket_numbers.append(int(match.group(1)))
        ticket_id = f"TKT{max(ticket_numbers, default=1000) + 1}"
        ticket = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "customer_name": customer.get("name"),
            "issue": issue,
            "status": "Open",
            "created_at": datetime.now().isoformat(),
        }
        tickets.append(ticket)
        save_bank_data(data)
        return (
            f"Support ticket {ticket_id} has been created successfully. "
            f"Issue: {issue}. Status: Open."
        )
    except Exception:
        return "Unable to create a support ticket right now."

account_tools = [
    read_account_details,
    raise_issue_ticket
]

credit_card_tools =[
    credit_card,
    outstanding_amount_pay
]

personal_tools =[
    read_personal_details,
    write_personal_details
]


def route_to_tools(state: AgentState):

    messages = state.get("messages", [])

    if not messages:
        return END

    last_message = messages[-1]

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:
            return "tools"

    return "start_message"


def route_agents(state: AgentState):
    """Dispatch selected agents; Send targets must be registered graph nodes."""
    agents = state.get("agents") or []
    if not isinstance(agents, list):
        agents = [str(agents)]
    if "NEEDS_CLARIFICATION" in agents:
        return [Send("clarification_agent", state)]

    valid_agents = {
        "ACCOUNT_AGENT": "account_agent",
        "CREDIT_CARD_AGENT": "credit_card_agent",
        "PERSONAL_INFORMATION_AGENT": "personal_detail_agent",
        "SECURITY_AGENT": "security_operations_agent",
        "AMOUNT_TRANSACTION_AGENT": "amount_transaction_agent",
    }
    routes = [valid_agents[name] for name in agents if name in valid_agents]
    if not routes:
        return [Send("clarification_agent", state)]
    return [Send(node_name, state) for node_name in routes]


def update_text_file(message_array):
    file_path = "messages.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        for message in message_array:
            file.write(f"{message}\n")

def greeting_agent(state: AgentState) -> dict:
    """Return a friendly greeting to the frontend without requiring customer data."""
    return {
        "messages": [AIMessage(content=(
            "Hello! How can I help you today? I can help with account details, "
            "credit-card information, personal information, security, and transactions."
        ))]
    }

def start_message(state: AgentState) -> dict:
    """Keep the frontend-provided request; never prompt in the terminal."""
    # The frontend sends `user_request` through the /chat route. If this graph is
    # invoked without a request, the router below sends it to the greeting node.
    return {"user_request": str(state.get("user_request", "") or "").strip()}



def check_user_details(state: AgentState) -> dict:
    """Validate a supplied customer ID without blocking on terminal input."""
    customer_id = str(state.get("customer_id", "") or "").strip()
    if not customer_id:
        return {
            "messages": [AIMessage(content=(
                "Please provide your customer ID so I can securely help with your banking request."
            ))]
        }

    try:
        data = load_bank_data()
        valid_ids = {str(c.get("customer_id", "")) for c in data.get("customers", [])}
    except (OSError, json.JSONDecodeError):
        return {
            "messages": [AIMessage(content=(
                "I can't verify customer details right now. Please try again later."
            ))]
        }

    if customer_id not in valid_ids:
        return {
            "customer_id": "",
            "messages": [AIMessage(content=(
                "I couldn't verify that customer ID. Please check it and try again."
            ))]
        }

    return {"customer_id": customer_id, "user_details": {"customer_id": customer_id}}


def is_greeting_request(user_request: str) -> bool:
    """Recognize standalone greetings so they do not trigger banking tools."""
    normalized = " ".join(
        str(user_request or "").lower().strip().strip(".!?,;:").split()
    )
    greetings = {
        "", "hi", "hello", "hey", "hello there", "hi there", "hey there",
        "good morning", "good afternoon", "good evening", "how are you",
        "how are you doing", "how can i help you today", "help",
    }
    return normalized in greetings


def route_after_start_message(state: AgentState) -> str:
    """Route greetings directly to a friendly response; other questions continue."""
    if is_greeting_request(state.get("user_request", "")):
        return "greeting_agent"
    return "check_user_details"


# Graph
app = StateGraph(AgentState)

app.add_node("start_message", start_message)
app.add_node("greeting_agent", greeting_agent)
app.add_node("check_user_details", check_user_details)
app.add_node("secrity_agent", secrity_agent)
app.add_node("orchestrator", orchestrator)
app.add_node("account_agent", account_agent)
app.add_node("credit_card_agent", credit_card_agent)
app.add_node("personal_detail_agent", personal_detail_agent)
app.add_node("security_operations_agent", security_operations_agent)
app.add_node("amount_transaction_agent", amount_transaction_agent)
app.add_node("clarification_agent", clarification_agent)

app.add_edge(START, "start_message")
app.add_conditional_edges(
    "start_message",
    route_after_start_message,
    {"greeting_agent": "greeting_agent", "check_user_details": "check_user_details"},
)
app.add_edge("greeting_agent", END)
app.add_conditional_edges(
    "check_user_details",
    orchestrator_2,
    {"security_agent": "secrity_agent", "end": END},
)
app.add_conditional_edges(
    "secrity_agent",
    orchestrator_1,
    {"orchestrator": "orchestrator", END: END},
)
app.add_conditional_edges("orchestrator", route_agents)

# Each agent executes only its own tool calls locally in run_tool_agent().
# This avoids unconditional ToolNode execution and cross-talk between parallel branches.
app.add_edge("account_agent", END)
app.add_edge("credit_card_agent", END)
app.add_edge("personal_detail_agent", END)
app.add_edge("security_operations_agent", END)
app.add_edge("amount_transaction_agent", END)
app.add_edge("clarification_agent", END)

graph = app.compile()

def main():
    initial_state = {
        "agents": [],
        "user_request": "",
        "messages": [],
        "security": "",
        "user_details": {},
        "customer_id": "",
    }
    result = graph.invoke(initial_state)
    print("\n--- Final response(s) ---")
    for message in result.get("messages", []):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            print(message.content)

if __name__ == "__main__":
    main()
