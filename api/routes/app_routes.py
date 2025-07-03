from fastapi import APIRouter
from fastapi import Query, Form
from api.services.app_services import spends_analyzer_service, budget_tool_services, get_raw_transactions_service, chat_old_service, chat_service
from api.utils.modals import ChatRequest, ChatResponse

router = APIRouter(prefix="", tags=["App Service"])

@router.get("/spends_analyzer")
async def spends_analyzer(
    customer_id: str = Query(...),
    use_flattened: bool = Query(False)
):
    return spends_analyzer_service(customer_id, use_flattened)

@router.get("/budget_tool")
async def budget_tool(
    customer_id: str = Query(...),
    use_flattened: bool = Query(False)
):
    return budget_tool_services(customer_id, use_flattened)

@router.get("/raw_transactions")
def get_raw_transactions(
    customer_id: str = Query(...),
    message: str = Query(None),
    use_flattened: bool = Query(False)
):
    return get_raw_transactions_service(customer_id, message, use_flattened)

@router.post("/chat_old")
async def chat_old(
    customer_id: str = Form(...),
    message: str = Form(...),
    use_flattened: bool = Form(False)
):
    return chat_old_service(customer_id, message, use_flattened)

@router.post("/chat", response_model=ChatResponse)
async def chat( request: ChatRequest ):
    return chat_service(request)