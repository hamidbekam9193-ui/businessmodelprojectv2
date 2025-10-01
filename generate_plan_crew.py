import os
import yaml
from crewai import Agent, Task, LLM

from tools.CharacterCounterTool import CharacterCounterTool

class PlanCrewFactory:
    """
    This class is a factory for creating the agents and tasks
    for the business plan generation crew.
    It is initialized with API keys to ensure all components are
    configured correctly from the start.
    """
    def __init__(self, gemini_api_key, groq_api_key):
        with open('config/agents.yaml', 'r') as f:
            self.agents_config = yaml.safe_load(f)
        with open('config/tasks.yaml', 'r') as f:
            self.tasks_config = yaml.safe_load(f)

        if not gemini_api_key:
            raise ValueError("A Gemini API key is required.")
        if not groq_api_key:
            raise ValueError("A Groq API key is required.")

        # --- DIAGNOSTIC PRINT STATEMENTS ---
        # This will show up in your Render logs to confirm the backend is receiving the keys.
        # We print only the last 4 characters for security.
        print(f"--- DIAGNOSTIC INFO ---")
        print(f"Received Gemini Key ending in: ...{gemini_api_key[-4:]}")
        print(f"Received Groq Key ending in:   ...{groq_api_key[-4:]}")
        print(f"-------------------------")
        # --- END DIAGNOSTIC ---

        # Set the environment variables that the underlying libraries are designed to look for.
        os.environ['GOOGLE_API_KEY'] = gemini_api_key
        os.environ['GROQ_API_KEY'] = groq_api_key

        # Initialize the LLMs. They will now automatically pick up the keys from the environment.
        self.llm_gemini = LLM(
            model="gemini-pro-latest"
        )
        
        self.llm_groq = LLM(
            model="groq/llama3-70b-8192"
        )

    # --- Methods to create each AGENT (No changes below this line) ---
    def business_designer(self):
        return Agent(
            config=self.agents_config['business_designer'],
            tools=[CharacterCounterTool()],
            llm=self.llm_gemini,
            verbose=True
        )

    def product_designer(self):
        return Agent(
            config=self.agents_config['product_designer'],
            tools=[CharacterCounterTool()],
            llm=self.llm_gemini,
            verbose=True
        )

    def market_analyst(self):
        return Agent(
            config=self.agents_config['market_analyst'],
            tools=[CharacterCounterTool()],
            llm=self.llm_gemini,
            verbose=True
        )

    def marketing_expert(self):
        return Agent(
            config=self.agents_config['marketing_expert'],
            tools=[CharacterCounterTool()],
            llm=self.llm_gemini,
            verbose=True
        )

    def operations_specialist(self):
        return Agent(
            config=self.agents_config['operations_specialist'],
            tools=[CharacterCounterTool()],
            llm=self.llm_gemini,
            verbose=True
        )

    def financial_expert(self):
        return Agent(
            config=self.agents_config['financial_expert'],
            tools=[CharacterCounterTool()],
            llm=self.llm_gemini,
            verbose=True
        )

    def consolidator(self):
        return Agent(
            config=self.agents_config['consolidator'],
            llm=self.llm_gemini,
            verbose=True
        )

    def evaluator(self):
        return Agent(
            config=self.agents_config['evaluator'],
            llm=self.llm_groq,
            verbose=True
        )

    def refiner(self):
        return Agent(
            config=self.agents_config['refiner'],
            llm=self.llm_groq,
            verbose=True
        )

    # --- Methods to create each TASK ---
    def create_business_concept_task(self, agent):
        return Task(config=self.tasks_config['create_business_concept'], agent=agent)

    def create_product_design_task(self, agent, context):
        return Task(config=self.tasks_config['create_product_design'], agent=agent, context=context)

    def create_market_analysis_task(self, agent, context):
        return Task(config=self.tasks_config['create_market_analysis'], agent=agent, context=context)

    def create_marketing_plan_task(self, agent, context):
        return Task(config=self.tasks_config['create_marketing_plan'], agent=agent, context=context)

    def create_operating_plan_task(self, agent, context):
        return Task(config=self.tasks_config['create_operating_plan'], agent=agent, context=context)

    def create_financial_plan_task(self, agent, context):
        return Task(config=self.tasks_config['create_financial_plan'], agent=agent, context=context)

    def consolidate_plan_task(self, agent, context):
        return Task(config=self.tasks_config['consolidate_plan'], agent=agent, context=context)

    def evaluate_plan_task(self, agent, context):
        return Task(config=self.tasks_config['evaluate_plan'], agent=agent, context=context)

    def refine_plan_task(self, agent, context):
        return Task(config=self.tasks_config['refine_plan'], agent=agent, context=context)


