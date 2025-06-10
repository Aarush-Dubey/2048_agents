# streamlit_app.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from agents import  RandomAgent, GreedyAgent, HeuristicAgent, ExpectimaxAgent
from game_engine import Game2048
from typing import List
from simulator import GameSimulator

def create_dashboard():
    st.set_page_config(page_title="2048 AI Dashboard", layout="wide")
    
    st.title("🎮 2048 AI Agent Dashboard")
    st.markdown("Compare different AI agents playing 2048")
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = GameSimulator()
    if 'agents' not in st.session_state:
        st.session_state.agents = {
            'Random': RandomAgent(),
            'Greedy': GreedyAgent(),
            'Heuristic': HeuristicAgent(),
            'Expectimax-3': ExpectimaxAgent(depth=3),
            'Expectimax-4': ExpectimaxAgent(depth=4)
        }
    
    # Sidebar controls
    st.sidebar.header("Simulation Settings")
    
    selected_agents = st.sidebar.multiselect(
        "Select Agents to Compare",
        options=list(st.session_state.agents.keys()),
        default=['Random', 'Heuristic']
    )
    
    num_games = st.sidebar.slider("Number of Games", 10, 1000, 100)
    
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running simulations..."):
            for agent_name in selected_agents:
                agent = st.session_state.agents[agent_name]
                st.session_state.simulator.simulate_agent(agent, num_games)
        st.sidebar.success("Simulation completed!")
    
    # Display results
    if selected_agents and any(name in st.session_state.simulator.results for name in selected_agents):
        display_results(st.session_state.simulator, selected_agents)
    else:
        st.info("Select agents and run simulation to see results")

def display_results(simulator: GameSimulator, selected_agents: List[str]):
    # Get statistics for all agents
    stats = {}
    for agent_name in selected_agents:
        if agent_name in simulator.results:
            stats[agent_name] = simulator.get_statistics(agent_name)
    
    if not stats:
        return
    
    # Summary metrics
    st.header("📊 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Average Score", 
                 f"{max(stats.values(), key=lambda x: x['score_mean'])['agent_name']}")
    
    with col2:
        st.metric("Highest Max Tile", 
                 f"{max(stats.values(), key=lambda x: x['max_tile_mean'])['agent_name']}")
    
    with col3:
        st.metric("Fastest Decision", 
                 f"{min(stats.values(), key=lambda x: x['avg_decision_time_ms'])['agent_name']}")
    
    with col4:
        st.metric("Most Moves", 
                 f"{max(stats.values(), key=lambda x: x['moves_mean'])['agent_name']}")
    
    # Detailed statistics table
    st.header("📈 Detailed Statistics")
    
    table_data = []
    for agent_name, stat in stats.items():
        table_data.append({
            'Agent': agent_name,
            'Avg Score': f"{stat['score_mean']:.0f}",
            'Max Score': f"{stat['score_max']:.0f}",
            'Avg Max Tile': f"{stat['max_tile_mean']:.0f}",
            'Avg Moves': f"{stat['moves_mean']:.0f}",
            'Decision Time (ms)': f"{stat['avg_decision_time_ms']:.1f}"
        })
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    
    # Visualizations
    st.header("📊 Visualizations")
    
    # Score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        fig_scores = go.Figure()
        
        for agent_name, stat in stats.items():
            scores = [r.score for r in stat['raw_results']]
            fig_scores.add_trace(go.Histogram(
                x=scores,
                name=agent_name,
                opacity=0.7,
                nbinsx=30
            ))
        
        fig_scores.update_layout(
            xaxis_title="Score",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        st.subheader("Max Tile Distribution")
        fig_tiles = go.Figure()
        
        for agent_name, stat in stats.items():
            max_tiles = [r.max_tile for r in stat['raw_results']]
            fig_tiles.add_trace(go.Histogram(
                x=max_tiles,
                name=agent_name,
                opacity=0.7
            ))
        
        fig_tiles.update_layout(
            xaxis_title="Max Tile",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_tiles, use_container_width=True)
    
    # Move frequency analysis
    st.subheader("Move Frequency Analysis")
    
    move_data = []
    for agent_name, stat in stats.items():
        total_moves = sum(stat['move_frequencies'].values())
        for move, count in stat['move_frequencies'].items():
            move_data.append({
                'Agent': agent_name,
                'Move': move,
                'Frequency': count / total_moves if total_moves > 0 else 0
            })
    
    if move_data:
        df_moves = pd.DataFrame(move_data)
        fig_moves = px.bar(df_moves, x='Move', y='Frequency', color='Agent',
                          title="Move Frequency by Agent", barmode='group')
        st.plotly_chart(fig_moves, use_container_width=True)
    
    # Performance vs Decision Time scatter
    st.subheader("Performance vs Decision Time")
    
    perf_data = []
    for agent_name, stat in stats.items():
        perf_data.append({
            'Agent': agent_name,
            'Avg Score': stat['score_mean'],
            'Decision Time (ms)': stat['avg_decision_time_ms'],
            'Avg Max Tile': stat['max_tile_mean']
        })
    
    df_perf = pd.DataFrame(perf_data)
    fig_perf = px.scatter(df_perf, x='Decision Time (ms)', y='Avg Score', 
                         size='Avg Max Tile', color='Agent',
                         title="Score vs Decision Time (bubble size = avg max tile)")
    st.plotly_chart(fig_perf, use_container_width=True)

if __name__ == "__main__":
    # For running the Streamlit app
    create_dashboard()