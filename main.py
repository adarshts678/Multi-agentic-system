import os
from typing import TypedDict , Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
import operator

import psycopg
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

DATABASE_URL=os.getenv("DATABASE_URL")
from langchain_google_genai import ChatGoogleGenerativeAI
llm=ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL"),
    temperature=0

)

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add]
    user_query: str
    flight_results: str
    hotel_results : str
    itinerary : str
    llm_calls: int


def flight_agent(state: TravelState) -> TravelState:
    query= state["user_query"]
    flight_results=search_flights(query)
    return {
        
        "messages": [AIMessage(content=flight_results)],
        "flight_results": flight_results,
        "llm_calls": state.get("llm_calls",0) + 1
    }

def hotel_agent(state:TravelState):
    query=state["user_query"]
    hotel_results=tavily_search(query)
    return{
        "messages":[AIMessage(content="hotel information fetched")],
        "hotel_results":hotel_results,
        "llm_calls": state.get("llm_calls",0)+1
    }

def scheduler_agent(state:TravelState):
    prompt=f"""
    you are a travel assistent.
    query:{state["user_query"]}
    flight_results: {state["flight_results"]}
    hotel_results:{state["hotel_results"]}

    """
    response = llm.invoke([
        SystemMessage(content="You are a travel planner"),
        HumanMessage(content=prompt)
    ])

    return {
        "messages": [response],   # response is already an AIMessage
        "itinerary": response.content,
        "llm_calls": state.get("llm_calls", 0) + 1
    }
def final_agent(state:TravelState):
    prompt=f"""
    give a final summary of all the agents results
    query:{state["user_query"]}
    flight: {state["flight_results"]}
    hotel:{state["hotel_results"]}
    itinerary:{state["itinerary"]}

    """
    response=llm.invoke([SystemMessage(content="you are a travel planner"),HumanMessage(content=prompt)])
    return {
        "messages":[AIMessage(content=response.content)],
        
        "llm_calls": state.get("llm_calls",0)+1
    }
state=StateGraph(TravelState)
state.add_node("flight_agent",flight_agent)
state.add_node("hotel_agent",hotel_agent)
state.add_node("scheduler_agent",scheduler_agent)
state.add_node("final_agent",final_agent)

state.add_edge(START,"flight_agent")
state.add_edge("flight_agent","hotel_agent")
state.add_edge("hotel_agent","scheduler_agent")
state.add_edge("scheduler_agent","final_agent")
state.add_edge("final_agent",END)

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app=state.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "user_adarsh"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)






