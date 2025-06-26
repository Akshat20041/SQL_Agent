from typing_extensions import TypedDict, Optional, Any, List
from langgraph.graph import StateGraph, START, END
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain_groq import ChatGroq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
import os
import logging
import re
import json

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(temperature=0, model="meta-llama/llama-4-scout-17b-16e-instruct")

# Initialize database
try:
    db = SQLDatabase.from_uri(
        "postgresql+psycopg2://postgres:{DB_PASS}@localhost:5432/postgres",
        schema="dc_ai_test"
    )
    logger.info("Database connected successfully")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    st.error(f"Failed to connect to database: {e}")
    raise

# Define State
class State(TypedDict):
    user_input: str
    sql_query: Optional[str]
    sql_result: Optional[Any]
    text_response: Optional[str]
    graph_data: Optional[Any]
    plots: Optional[List[Any]]  # Changed to support multiple plots
    explanation: Optional[str]
    final_ans: Optional[str]
    graph_suggestions: Optional[List[str]]  # New field for graph suggestions

# Initialize SQL agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

def NLP_Query(state: State) -> State:
    """Convert natural language query to SQL."""
    try:
        prompt = f"""
        You are a SQL assistant. Convert the following natural language query into a pure SQL SELECT statement.
        ONLY return the SQL query. DO NOT explain anything, DO NOT add any text.
        
        Query: {state['user_input']}
        Schema: dc_ai_test
        """
        response = llm.invoke(prompt)
        sql_query = response.content.strip().strip("```sql").strip("```")
        logger.info(f"Generated SQL query: {sql_query}")
        return {"sql_query": sql_query}
    except Exception as e:
        logger.error(f"Error in NLP_Query: {e}")
        return {"sql_query": None, "text_response": f"Error generating SQL query: {e}"}

def SQL_Agent(state: State) -> State:
    """Execute SQL query using the agent executor."""
    try:
        if not state["sql_query"]:
            raise ValueError("No SQL query provided")
        
        # First try to get structured data directly from database
        try:
            df = pd.read_sql(state["sql_query"], db._engine)
            logger.info(f"Direct SQL execution successful. DataFrame shape: {df.shape}")
            return {"sql_result": df}
        except Exception as direct_error:
            logger.warning(f"Direct SQL execution failed: {direct_error}")
            # Fallback to agent executor
            result = agent_executor.invoke({"input": state["sql_query"]})
            sql_result = result.get("output", [])
            logger.info(f"Agent SQL result: {sql_result}")
            return {"sql_result": sql_result}
    except Exception as e:
        logger.error(f"Error in SQL_Agent: {e}")
        return {"sql_result": None, "text_response": f"Error executing SQL query: {e}"}

def Text_Response(state: State) -> State:
    """Convert SQL result to text response."""
    try:
        rows = state["sql_result"]
        if rows is None:
            return {"text_response": "No results found"}
        
        # Handle DataFrame directly
        if isinstance(rows, pd.DataFrame):
            if rows.empty:
                return {"text_response": "No results found"}
            text = f"Found {len(rows)} records:\n"
            text += rows.to_string(index=False, max_rows=10)
            if len(rows) > 10:
                text += f"\n... and {len(rows) - 10} more rows"
        elif isinstance(rows, list):
            text = "\n".join([f"{row[0]} - {row[1]}" if isinstance(row, (list, tuple)) and len(row) >= 2 else str(row) for row in rows])
        elif isinstance(rows, str):
            text = rows
        else:
            text = str(rows)
        
        logger.info(f"Text response: {text[:200]}...")
        return {"text_response": text}
    except Exception as e:
        logger.error(f"Error in Text_Response: {e}")
        return {"text_response": f"Error processing SQL result: {e}"}

def parse_agent_output_to_dataframe(agent_output: str) -> pd.DataFrame:
    """Parse agent output string into DataFrame."""
    try:
        # Common patterns in agent outputs
        lines = agent_output.strip().split('\n')
        data = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('Action:') or line.startswith('Thought:') or line.startswith('Observation:'):
                continue
                
            # Try to parse different formats
            if ' - ' in line:
                parts = line.split(' - ', 1)
                if len(parts) == 2:
                    item, value = parts
                    try:
                        value = float(value.replace(',', '').replace('$', ''))
                        data.append({'item': item.strip(), 'value': value})
                    except:
                        data.append({'item': item.strip(), 'value': value.strip()})
            elif '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 2:
                    data.append({'item': parts[0], 'value': parts[1]})
        
        if data:
            return pd.DataFrame(data)
        else:
            # Try to extract numbers and text
            import re
            pattern = r'([A-Za-z\s]+)\s*[\-\:]\s*([0-9\.,]+)'
            matches = re.findall(pattern, agent_output)
            if matches:
                data = [{'item': item.strip(), 'value': float(value.replace(',', ''))} for item, value in matches]
                return pd.DataFrame(data)
            
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error parsing agent output: {e}")
        return pd.DataFrame()

def suggest_graph_types(query: str, df: pd.DataFrame) -> List[str]:
    """Suggest multiple graph types based on query and data."""
    suggestions = []
    query_lower = query.lower()
    
    if df.empty:
        return suggestions
    
    # Check what user specifically requested
    requested_charts = []
    if "pie chart" in query_lower or "pie" in query_lower:
        requested_charts.append("pie")
    if "bar chart" in query_lower or "bar" in query_lower:
        requested_charts.append("bar")
    if "line chart" in query_lower or "line" in query_lower:
        requested_charts.append("line")
    if "scatter" in query_lower:
        requested_charts.append("scatter")
    if "histogram" in query_lower:
        requested_charts.append("histogram")
    
    # If user specifically requested charts, prioritize those
    if requested_charts:
        suggestions.extend(requested_charts)
    
    # Add contextual suggestions based on data
    num_cols = df.shape[1]
    has_numeric = any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
    has_categorical = any(pd.api.types.is_string_dtype(df[col]) for col in df.columns)
    
    if num_cols >= 2 and has_numeric and has_categorical:
        if "bar" not in suggestions:
            suggestions.append("bar")
        if "pie" not in suggestions and "top" in query_lower:
            suggestions.append("pie")
    
    # Add alternative views
    if len(df) > 1 and has_numeric:
        if "line" not in suggestions and any(word in query_lower for word in ["trend", "over time", "month", "year"]):
            suggestions.append("line")
        if "histogram" not in suggestions and "distribution" in query_lower:
            suggestions.append("histogram")
    
    # If no suggestions, add default based on data structure
    if not suggestions:
        if num_cols >= 2:
            suggestions.append("bar")
        if has_categorical and has_numeric:
            suggestions.append("pie")
    
    return suggestions[:3]  # Limit to 3 suggestions

def create_graph(df: pd.DataFrame, graph_type: str, query: str) -> go.Figure:
    """Create a specific type of graph from DataFrame."""
    try:
        if df.empty:
            return None
            
        # Clean column names
        df.columns = [re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_') for col in df.columns]
        
        # Get the best columns for plotting
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
        
        if graph_type == "pie":
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                fig = px.pie(df, names=categorical_cols[0], values=numeric_cols[0],
                           title=f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}")
            elif len(categorical_cols) >= 1:
                fig = px.pie(df, names=categorical_cols[0], title=f"Distribution of {categorical_cols[0]}")
            else:
                return None
        
        elif graph_type == "bar":
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                           title=f"{numeric_cols[0]} by {categorical_cols[0]}")
                fig.update_layout(xaxis_tickangle=-45)
            else:
                return None
        
        elif graph_type == "line":
            if len(df.columns) >= 2:
                fig = px.line(df, x=df.columns[0], y=df.columns[1],
                            title=f"Trend of {df.columns[1]} over {df.columns[0]}")
            else:
                return None
        
        elif graph_type == "scatter":
            if len(numeric_cols) >= 2:
                color_col = categorical_cols[0] if categorical_cols else None
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=color_col,
                               title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
            else:
                return None
        
        elif graph_type == "histogram":
            if len(numeric_cols) >= 1:
                fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
            else:
                return None
        
        else:
            return None
        
        # Customize layout
        fig.update_layout(
            showlegend=True,
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
            font=dict(size=12)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating {graph_type} graph: {e}")
        return None

def Graphical_Response(state: State) -> State:
    """Generate multiple plots from SQL result intelligently."""
    try:
        rows = state["sql_result"]
        logger.info(f"Raw SQL result type: {type(rows)}")
        
        if rows is None:
            logger.warning("No SQL result for plotting")
            return {"graph_data": None, "plots": [], "graph_suggestions": []}
        
        # Convert to DataFrame
        df = None
        if isinstance(rows, pd.DataFrame):
            df = rows.copy()
        elif isinstance(rows, str):
            logger.info("Parsing string result to DataFrame")
            df = parse_agent_output_to_dataframe(rows)
        elif isinstance(rows, list) and len(rows) > 0:
            if isinstance(rows[0], dict):
                df = pd.DataFrame(rows)
            else:
                # Try to create DataFrame from list
                df = pd.DataFrame(rows)
        
        if df is None or df.empty:
            logger.warning("Could not create DataFrame from SQL result")
            return {"graph_data": None, "plots": [], "graph_suggestions": []}
        
        logger.info(f"DataFrame created with shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # Get graph suggestions
        suggestions = suggest_graph_types(state["user_input"], df)
        logger.info(f"Graph suggestions: {suggestions}")
        
        # Create multiple plots
        plots = []
        for graph_type in suggestions:
            fig = create_graph(df, graph_type, state["user_input"])
            if fig:
                plots.append({
                    "type": graph_type,
                    "figure": fig,
                    "title": f"{graph_type.title()} Chart"
                })
        
        # If no plots created, try a simple bar chart as fallback
        if not plots and len(df.columns) >= 2:
            try:
                fig = px.bar(df.head(10), x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                           title="Data Visualization")
                fig.update_layout(height=400)
                plots.append({
                    "type": "bar",
                    "figure": fig,
                    "title": "Bar Chart (Fallback)"
                })
            except:
                pass
        
        logger.info(f"Created {len(plots)} plots")
        return {
            "graph_data": df.to_dict(orient="records"),
            "plots": plots,
            "graph_suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Error in Graphical_Response: {e}")
        return {"graph_data": None, "plots": [], "graph_suggestions": []}

def Merge_Response(state: State) -> State:
    """Combine text, explanation, and graph into final answer."""
    try:
        # Enhanced prompt to suggest multiple visualizations
        prompt = f"""
        Generate a comprehensive final answer based on the following:
        User Query: {state['user_input']}
        Text Response: {state['text_response']}
        Available Graphs: {state.get('graph_suggestions', [])}
        
        Provide a detailed answer that:
        1. Directly answers the user's question
        2. Highlights key findings from the data
        3. Mentions the available visualizations
        4. Provides insights and recommendations
        """
        response = llm.invoke(prompt)
        final_ans = response.content
        logger.info(f"Final answer generated: {final_ans[:100]}...")
        return {
            "final_ans": final_ans,
            "plots": state.get("plots", [])
        }
    except Exception as e:
        logger.error(f"Error in Merge_Response: {e}")
        return {
            "final_ans": f"Error generating final answer: {e}",
            "plots": state.get("plots", [])
        }

# Build and compile graph
graph = StateGraph(State)
graph.add_node("NLP_Query", NLP_Query)
graph.add_node("SQL_Agent", SQL_Agent)
graph.add_node("Text_Response", Text_Response)
graph.add_node("Graphical_Response", Graphical_Response)
graph.add_node("Merge_Response", Merge_Response)

graph.add_edge(START, "NLP_Query")
graph.add_edge("NLP_Query", "SQL_Agent")
graph.add_edge("SQL_Agent", "Text_Response")
graph.add_edge("SQL_Agent", "Graphical_Response")
graph.add_edge("Text_Response", "Merge_Response")
graph.add_edge("Graphical_Response", "Merge_Response")
graph.add_edge("Merge_Response", END)

app = graph.compile()

# Streamlit app with session state
st.title("🔍 SQL Query Assistant with Multiple Intelligent Graphs")
st.write("Enter a query to retrieve data and generate multiple visualizations automatically.")

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None

# Input query
user_input = st.text_input("Enter your query:",
                          value="list me top 10 items that were most sold in the month of May ? Also Show this in the form of pie chart",
                          help="Try: 'Show sales by category with bar and pie charts' or 'Top products with multiple visualizations'")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Run Query", type="primary"):
        if user_input:
            with st.spinner("Processing your query and generating visualizations..."):
                try:
                    state = {"user_input": user_input}
                    result = app.invoke(state)
                    st.session_state.result = result
                    st.success("Query processed successfully!")
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    logger.error(f"Streamlit error: {e}")
        else:
            st.warning("Please enter a query")

with col2:
    if st.button("🔄 Clear Results"):
        st.session_state.result = None
        st.success("Results cleared!")

# Display results if available
if st.session_state.result:
    result = st.session_state.result
    
    # Main results
    st.subheader("📊 Analysis Results")
    st.write(result["final_ans"])
    
    # Multiple Graphs Section
    if result.get("plots") and len(result["plots"]) > 0:
        st.subheader("📈 Data Visualizations")
        
        # Create tabs for different graph types
        plot_tabs = st.tabs([f"{plot['title']}" for plot in result["plots"]])
        
        for i, (tab, plot) in enumerate(zip(plot_tabs, result["plots"])):
            with tab:
                st.plotly_chart(plot["figure"], use_container_width=True)
                st.caption(f"Graph Type: {plot['type'].title()}")
        
    # Additional Information
    with st.expander("🔍 Query Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Generated SQL")
            st.code(result.get("sql_query", "No query generated"), language="sql")
        with col2:
            st.subheader("Raw Data Sample")
            if result.get("graph_data"):
                df_display = pd.DataFrame(result["graph_data"])
                st.dataframe(df_display.head(10), use_container_width=True)
            else:
                st.write("No structured data available")
    
    # Debug Information
    with st.expander("🛠️ Debug Information"):
        st.write("**SQL Result Type:**", type(result.get("sql_result")))
        st.write("**Graph Suggestions:**", result.get("graph_suggestions", []))
        st.write("**Number of Plots Generated:**", len(result.get("plots", [])))
        if result.get("text_response"):
            st.write("**Text Response:**")
            st.text(result["text_response"][:500] + "..." if len(result["text_response"]) > 500 else result["text_response"])

