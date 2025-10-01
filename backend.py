from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List,  Optional
import os
import traceback
import google.generativeai as genai
from dotenv import load_dotenv

from main import BusinessPlanFlow, BusinessPlanState

app = FastAPI()

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

class BusinessPlanRequest(BaseModel):
    business_name: str
    start_year: str
    business_reason: str
    mission_vision: str
    legal_structure: str
    financial_funding: List[str]
    business_sector: str
    product_service_description: str
    
    # Business Sector Information
    raw_materials_type: Optional[str] = ""
    industrial_business_type: Optional[str] = ""
    services_type: Optional[str] = ""
    durable_goods_type: Optional[str] = ""
    consumer_goods_type: Optional[str] = ""
    healthcare_type: Optional[str] = ""
    financial_sector_type: Optional[str] = ""
    it_sector_type: Optional[str] = ""
    utilities_type: Optional[str] = ""
    culture_type: Optional[str] = ""
    
    # Market Information
    primary_countries: str
    product_centralisation: str
    product_range: str
    end_consumer_characteristics: str
    end_consumer_characteristics_2: List[str]

    # Segmentation Information
    segment_name: str
    segment_demographics: str
    segment_characteristics: str
    customer_count: str
    problems_faced: str
    biggest_competitors: str
    competition_intensity: str
    price_comparison: str
    market_type: str
    competitive_parameters: List[str]
    value_propositions: List[str]
    direct_income: str
    primary_revenue: List[str]
    one_time_payments: Optional[List[str]] = []
    ongoing_payments: Optional[List[str]] = []
    payment_characteristics: Optional[List[str]] = []
    package_price: str
    price_negotiation: str
    fixed_prices: Optional[List[str]] = []
    dynamic_prices: Optional[List[str]] = []
    distribution_channels: List[str]
    purchasing_power: str
    product_related_characteristics: List[str]
    self_service_availability: str
    online_communities_presence: str
    development_process_customer_involvement: str
    after_sale_purchases: str
    personal_assistance_offered: str
    similar_products_switch: str
    general_customer_relation: str

    # Key resources
    material_resources: List[str]
    intangible_resources: List[str]
    important_activities: List[str]
    inhouse_activities: List[str]
    outsourced_activities: Optional[List[str]] = []

    # Company statements
    company_statements: Optional[List[str]] = []
    
    # Important strategic partners
    important_strategic_partners: List[str]
    partnership_benefits: List[str]
    other_benefit: Optional[str] = ""
    company_dependency: str
    cost_intensive_components: List[str]

    # Team
    team_members: str
    funding_amount: str
    funding_purpose: str

class BusinessPlanResponse(BaseModel):
    business_plan: str

def collect_business_plan_inputs(request: BusinessPlanRequest) -> dict:
    """
    Collects and formats the inputs from the frontend request for use in the crew.
    
    Args:
        request (BusinessPlanRequest): The request object containing all user inputs
        
    Returns:
        dict: A dictionary containing all the inputs formatted for use in the crew
    """
    def safe_join(lst):
        """Safely join a list into a string, handling None and empty lists."""
        if lst is None:
            return ""
        return ", ".join(lst) if lst else ""

    inputs = {
        "business_name": request.business_name,
        "start_year": request.start_year,
        "business_reason": request.business_reason,
        "mission_vision": request.mission_vision,
        "legal_structure": request.legal_structure,
        "financial_funding": safe_join(request.financial_funding),
        "business_sector": request.business_sector,
        # Business Sector Information
        "raw_materials_type": request.raw_materials_type,
        "industrial_business_type": request.industrial_business_type,
        "services_type": request.services_type,
        "durable_goods_type": request.durable_goods_type,
        "consumer_goods_type": request.consumer_goods_type,
        "healthcare_type": request.healthcare_type,
        "financial_sector_type": request.financial_sector_type,
        "it_sector_type": request.it_sector_type,
        "utilities_type": request.utilities_type,
        "culture_type": request.culture_type,
        "product_service_description": request.product_service_description,
        "primary_countries": request.primary_countries,
        "product_centralisation": request.product_centralisation,
        "product_range": request.product_range,
        "end_consumer_characteristics": request.end_consumer_characteristics,
        "end_consumer_characteristics_2": safe_join(request.end_consumer_characteristics_2),
        "segment_name": request.segment_name,
        "segment_demographics": request.segment_demographics,
        "segment_characteristics": request.segment_characteristics,
        "customer_count": request.customer_count,
        "problems_faced": request.problems_faced,
        "biggest_competitors": request.biggest_competitors,
        "competition_intensity": request.competition_intensity,
        "price_comparison": request.price_comparison,
        "market_type": request.market_type,
        "competitive_parameters": safe_join(request.competitive_parameters),
        "value_propositions": safe_join(request.value_propositions),
        "direct_income": request.direct_income,
        "primary_revenue": safe_join(request.primary_revenue),
        # Optional payment and pricing information
        "one_time_payments": safe_join(request.one_time_payments),
        "ongoing_payments": safe_join(request.ongoing_payments),
        "payment_characteristics": safe_join(request.payment_characteristics),
        "package_price": request.package_price,
        "price_negotiation": request.price_negotiation,
        "fixed_prices": safe_join(request.fixed_prices),
        "dynamic_prices": safe_join(request.dynamic_prices),
        "distribution_channels": safe_join(request.distribution_channels),
        "purchasing_power": request.purchasing_power,
        "product_related_characteristics": safe_join(request.product_related_characteristics),
        "self_service_availability": request.self_service_availability,
        "online_communities_presence": request.online_communities_presence,
        "development_process_customer_involvement": request.development_process_customer_involvement,
        "after_sale_purchases": request.after_sale_purchases,
        "personal_assistance_offered": request.personal_assistance_offered,
        "similar_products_switch": request.similar_products_switch,
        "general_customer_relation": request.general_customer_relation,
        "material_resources": safe_join(request.material_resources),
        "intangible_resources": safe_join(request.intangible_resources),
        "important_activities": safe_join(request.important_activities),
        "inhouse_activities": safe_join(request.inhouse_activities),
        "outsourced_activities": safe_join(request.outsourced_activities),
        "company_statements": safe_join(request.company_statements),
        "important_strategic_partners": safe_join(request.important_strategic_partners),
        "partnership_benefits": safe_join(request.partnership_benefits),
        "other_benefit": request.other_benefit,
        "company_dependency": request.company_dependency,
        "cost_intensive_components": safe_join(request.cost_intensive_components),
        "team_members": request.team_members,
        "funding_amount": request.funding_amount,
        "funding_purpose": request.funding_purpose
    }

    return inputs

@app.post("/generate_business_plan", response_model=BusinessPlanResponse)
async def generate_business_plan(request: BusinessPlanRequest):
    try:
        inputs = collect_business_plan_inputs(request)

        initial_state = BusinessPlanState(user_inputs=inputs)

        flow = BusinessPlanFlow()

        state_result = await flow.kickoff_async(initial_state.model_dump())

        if isinstance(state_result, BaseModel):
            state_dict = state_result.model_dump()
        elif isinstance(state_result, dict):
            state_dict = state_result
        else:
            raise ValueError("Flow result is not a dict or Pydantic model")

        bp_value = state_dict.get("business_plan")
        if isinstance(bp_value, dict) and "raw" in bp_value:
            state_dict["business_plan"] = bp_value["raw"]

        final_state = BusinessPlanState(**state_dict)

        return BusinessPlanResponse(business_plan=final_state.business_plan)
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

