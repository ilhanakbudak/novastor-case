"""Tools for the NovaStor assistant.

Two families of tool:
  1. Knowledge search — retrieval over the document knowledge base (RAG).
  2. Account tools — lookups into operational data.

SECURITY DESIGN (the key decision in this case):
The account tools are built PER REQUEST with the caller's `customer_id` already
bound in (a closure). The language model therefore cannot choose whose account
to read — `get_account_summary()` takes no customer argument, and
`get_shipment_status(shipment_id)` is checked against the bound customer. Even
if a user asks to see another company's data, no tool can produce it. Identity
used for authorization must come from the session, never from the model.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever

from app.data.operations import OperationsStore
from app.exceptions import NotFoundError

logger = logging.getLogger(__name__)


def _format_citation(metadata: dict) -> str:
    src = metadata.get("source", "?")
    page = metadata.get("page")
    return f"{src} p.{page}" if page else src


def build_tools(customer_id: str, retriever: VectorStoreRetriever, ops: OperationsStore):
    """Build the toolset for one authenticated customer.

    customer_id is closed over and never exposed as a model-supplied argument.
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search NovaStor's documentation (services, policies, SLAs, pricing,
        contract terms) for passages relevant to the query. Use this for any
        general question about how NovaStor's services work."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the knowledge base."
        return "\n\n".join(
            f"[{_format_citation(d.metadata)}] {d.page_content}" for d in docs
        )

    @tool
    def get_account_summary() -> str:
        """Get the calling customer's own account: contract tier, contract dates,
        storage quota and usage, outstanding invoice, and account manager."""
        a = ops.get_account(customer_id)  # bound identity — not model-controlled
        used, quota = a["storage_used_m3"], a["storage_quota_m3"]
        pct = round(100 * used / quota) if quota else 0
        return (
            f"Company: {a['company_name']} | Tier: {a['contract_tier']}\n"
            f"Contract: {a['contract_start']} to {a['contract_end']}\n"
            f"Storage: {used} / {quota} m3 ({pct}% of quota)\n"
            f"Climate-controlled: {a['climate_controlled']}\n"
            f"Outstanding invoice: EUR {a['outstanding_invoice_eur']} "
            f"({a['payment_status']})\n"
            f"Account manager: {a['account_manager']}"
        )

    @tool
    def get_shipment_status(shipment_id: str) -> str:
        """Get the status, route, and ETA of one of the calling customer's
        shipments, by its shipment id (e.g. 'SH-1042')."""
        try:
            s = ops.get_shipment(shipment_id, customer_id)
        except NotFoundError as exc:
            return exc.message  # safe message; never another customer's data
        eta = s["eta"] or "not scheduled"
        return (
            f"Shipment {s['shipment_id']}: {s['status']} | "
            f"{s['origin']} -> {s['destination']} | items: {s['items']} | ETA: {eta}"
        )

    @tool
    def list_my_shipments() -> str:
        """List all shipments on the calling customer's account with their status."""
        shipments = ops.list_shipments(customer_id)
        if not shipments:
            return "You have no shipments on record."
        return "\n".join(
            f"{s['shipment_id']}: {s['status']} ({s['origin']} -> {s['destination']})"
            for s in shipments
        )

    return [search_knowledge_base, get_account_summary, get_shipment_status, list_my_shipments]
