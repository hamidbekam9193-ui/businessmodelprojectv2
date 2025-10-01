#!/usr/bin/env python
from typing import Dict
from pydantic import BaseModel
from crewai import Crew, Process
from crewai.flow import Flow, start

# Import the new factory class
from generate_plan_crew import PlanCrewFactory

class BusinessPlanState(BaseModel):
    user_inputs: Dict = {}
    business_plan: str = ""

class BusinessPlanFlow(Flow[BusinessPlanState]):
    @start("done")
    def generate_business_plan(self):
        print("Starting business plan generation flow...")
        inputs = self.state.user_inputs
        gemini_api_key = inputs.pop('gemini_api_key', None)
        groq_api_key = inputs.pop('groq_api_key', None)

        # Use the factory to create all necessary components
        factory = PlanCrewFactory(gemini_api_key, groq_api_key)

        # 1. Instantiate all agents
        business_designer_agent = factory.business_designer()
        product_designer_agent = factory.product_designer()
        market_analyst_agent = factory.market_analyst()
        marketing_expert_agent = factory.marketing_expert()
        operations_specialist_agent = factory.operations_specialist()
        financial_expert_agent = factory.financial_expert()
        consolidator_agent = factory.consolidator()
        evaluator_agent = factory.evaluator()
        refiner_agent = factory.refiner()

        # 2. Instantiate all tasks, linking them to agents and context
        task1 = factory.create_business_concept_task(business_designer_agent)
        task2 = factory.create_product_design_task(product_designer_agent, context=[task1])
        task3 = factory.create_market_analysis_task(market_analyst_agent, context=[task1, task2])
        task4 = factory.create_marketing_plan_task(marketing_expert_agent, context=[task1, task2, task3])
        task5 = factory.create_operating_plan_task(operations_specialist_agent, context=[task1, task2, task3, task4])
        task6 = factory.create_financial_plan_task(financial_expert_agent, context=[task1, task2, task3, task4, task5])
        task7 = factory.consolidate_plan_task(consolidator_agent, context=[task1, task2, task3, task4, task5, task6])
        task8 = factory.evaluate_plan_task(evaluator_agent, context=[task7])
        task9 = factory.refine_plan_task(refiner_agent, context=[task7, task8])

        # 3. Manually assemble the Crew
        business_plan_crew = Crew(
            agents=[
                business_designer_agent,
                product_designer_agent,
                market_analyst_agent,
                marketing_expert_agent,
                operations_specialist_agent,
                financial_expert_agent,
                consolidator_agent,
                evaluator_agent,
                refiner_agent,
            ],
            tasks=[task1, task2, task3, task4, task5, task6, task7, task8, task9],
            process=Process.sequential,
            verbose=2
        )

        # 4. Kickoff the crew with the remaining user inputs
        print("Kicking off the manually assembled crew...")
        result = business_plan_crew.kickoff(inputs=inputs)
        
        self.state.business_plan = result
        print("Crew kickoff complete. Result has been stored.")
        return self.state

    def done(self):
        return self.state
