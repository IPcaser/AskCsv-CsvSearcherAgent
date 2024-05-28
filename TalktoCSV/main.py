from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import CSVSearchTool
import os
os.environ["OPENAI_API_KEY"] = "NA"
llm = ChatOpenAI(
    model="crewai-llama3",
    base_url="http://localhost:11434/v1"
)

tool = CSVSearchTool(
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="llama3",
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="sentence-transformers/msmarco-distilbert-base-v4"
                
            ),
        ),
    ),
   csv="D:/MLprojects/CSVbot/sample.csv"
)

Analyser=Agent(
    role='Data analyst',
    goal='To Generate valuable insights in the CSV data',
    backstory="""You are a professional and highly skilled data analyst.
    Your role is to provide the valuable insights on the users bussiness or local data and provide with best analysis results.
    You should provide the most accurate and consistent results""",
    allow_delegation=False,
    llm=llm,
    tools=[tool],
    verbose=True
)

Report_Generator=Agent(
    role='Report and insights writer',
    goal='To generate the reports on the data insights',
    backstory="""You are a professional and highly skilled report writer.
    Your role is to generate the reports on the users data and requirement.
    You should provide the most accurate and consistent results""",
    allow_delegation=True,
    llm=llm,
    verbose=True
)

# task1=Task(
#     description='Generate insights on test and average score with the topper',
#     agent=Analyser,
#     expected_output='Summary of the data'
# )

task2 = Task(
  description="""Generate report on the analyzed data. 
  **Data:** This report should be based on the data in the CSV file located at D:/MLprojects/CSVbot/sample.csv. 
  **Scope:** Please identify trends, patterns, and correlations in the data and focus on the most important insights. 
  **Expected Output:** A brief summary of the results.""",
  agent=Report_Generator,
  expected_output="A brief summary of the results"
)

mycrew=Crew(
    tasks=[task2],
    agents=[Analyser,Report_Generator],
    verbose=True
)

results=mycrew.kickoff()

print(results)