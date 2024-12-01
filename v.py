import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from SBTi.data.excel import ExcelProvider
from SBTi.temperature_score import TemperatureScore
from SBTi.portfolio_aggregation import PortfolioAggregationMethod
from SBTi.interfaces import ETimeFrames, EScope, PortfolioCompany
from typing import List
import numpy as np
import plotly.express as px

def calculate_emissions_reduction(base_year_ghg_s1, base_year_ghg_s2, base_year_ghg_s3,
                                  reduction_ambition, start_year, end_year):
    # Generate the years
    years = np.arange(start_year, end_year + 1)
    num_years = len(years)

    # Ensure we don't have division by zero or negative values
    reduction_ambition = min(max(reduction_ambition, 0), 1)  # Ensure between 0 and 1

    # Calculate linear reduction for each year
    reduction_factor = np.linspace(0, reduction_ambition, num_years)

    # Calculate emissions for each year
    emissions_s1 = base_year_ghg_s1 * (1 - reduction_factor)
    emissions_s2 = base_year_ghg_s2 * (1 - reduction_factor)
    emissions_s3 = base_year_ghg_s3 * (1 - reduction_factor)

    # Calculate total emissions
    total_emissions = emissions_s1 + emissions_s2 + emissions_s3
    base_total_emissions = base_year_ghg_s1 + base_year_ghg_s2 + base_year_ghg_s3

    # Calculate temperature scores
    initial_temp_score = 3.2  # As per the SBTi result
    min_temp_score = 1.5  # Paris Agreement goal

    # Calculate temperature score reduction based on emission reduction
    temp_score_range = initial_temp_score - min_temp_score
    emission_reduction_fraction = 1 - (total_emissions / base_total_emissions)
    temperature_scores = initial_temp_score - (temp_score_range * emission_reduction_fraction)

    # Ensure temperature scores don't go below the minimum
    temperature_scores = np.maximum(temperature_scores, min_temp_score)

    # Create DataFrame for results
    emissions_df = pd.DataFrame({
        "Year": years,
        "S1 Emissions": emissions_s1,
        "S2 Emissions": emissions_s2,
        "S3 Emissions": emissions_s3,
        "Total Emissions": total_emissions,
        "Temperature Score": temperature_scores
    })

    return emissions_df

def create_app():
    st.set_page_config(page_title="Temperature Score Analysis", layout="wide")

    # Sidebar Configuration
    with st.sidebar:
        st.image("https://drive.google.com/file/d/1dYFtHKPfIRamXTFdrAGkAfiv7KRxgMM8/view?usp=sharing")
        st.markdown("<h2 style='display:inline; font-family:Arial;'>Verdure Climate</h2>", unsafe_allow_html=True)

        st.button("Click here if you have issues", help="Learn how to use the app")
        with st.expander("How to use the app"):
            st.markdown("""
                - **Upload Portfolio and Data Provider files** in the respective sections.
                - View the **data previews** to ensure correctness.
                - Analyze **Temperature Scores** and visualizations.
                - Download the results as a CSV file.

                **Required Documents**:
                1. Portfolio Excel File
                2. Data Provider Excel File

                **Structure**:
                - Upload Data
                - Data Previews
                - Emission Reduction Tables
                - Visualizations (Pie Chart, Trends)
                """)

    # Add title and description
    st.title("Portfolio Temperature Score Analysis")
    st.markdown("""
    This application analyzes portfolio temperature scores using the SBTi methodology.
    Upload your portfolio and data provider files to get started.
    """)

    # File upload section
    st.header("Upload Files")
    col1, col2 = st.columns(2)

    with col1:
        portfolio_file = st.file_uploader("Upload Portfolio Excel File", type=['xlsx'])
    with col2:
        data_provider_file = st.file_uploader("Upload Data Provider Excel File", type=['xlsx'])

    if portfolio_file and data_provider_file:
        try:
            # Load data
            df_portfolio = pd.read_excel(portfolio_file, sheet_name="in")
            df_portfolio = df_portfolio.set_index('company_id', verify_integrity=True)

            # Load data provider information
            provider = ExcelProvider(path=data_provider_file)
            target_data = pd.read_excel(data_provider_file, sheet_name="target_data")

            # Add debugging information
            st.write("Number of rows in target_data:", len(target_data))
            st.write("Columns in target_data:", target_data.columns.tolist())
            st.write("First few rows of target_data:")
            st.write(target_data.head())

            # Show data preview
            st.header("Data Preview")
            st.subheader("Portfolio Data")
            st.dataframe(df_portfolio.head())

            st.subheader("Data Provider Target Data")
            st.dataframe(target_data.head())

            # Process data
            portfolio_companies = []
            for index, row in df_portfolio.iterrows():
                company = PortfolioCompany(
                    company_name=row['company_name'],
                    company_id=str(index),  # Convert to string
                    investment_value=row['investment_value'],
                    company_isin=row['company_isin'],
                    user_fields={}
                )
                portfolio_companies.append(company)

            # Calculate scores
            temperature_score = TemperatureScore(
                time_frames=list(ETimeFrames),
                scopes=[EScope.S1S2, EScope.S3, EScope.S1S2S3],
                aggregation_method=PortfolioAggregationMethod.WATS
            )

            with st.spinner("Calculating temperature scores..."):
                amended_portfolio = temperature_score.calculate(
                    data_providers=[provider],
                    portfolio=portfolio_companies
                )
                aggregated_scores = temperature_score.aggregate_scores(amended_portfolio)
                scores_df = pd.DataFrame(aggregated_scores.dict())
                scores_df = scores_df.applymap(lambda x: round(x['all']['score'], 2)
                if isinstance(x, dict) and 'all' in x and 'score' in x['all']
                else x)

            # Calculate emission reductions and predicted temperature scores for each company
            emission_results = []
            for index, row in df_portfolio.iterrows():
                company_name = row['company_name']
                company_id = str(index)  # Convert to string to ensure matching

                # Fetch all rows for the company_id from target_data
                company_targets = target_data[target_data['company_id'].astype(str) == company_id]

                if company_targets.empty:
                    st.warning(f"No targets found for company: {company_name} (ID: {company_id})")
                    continue

                # Initialize variables with default values
                base_year_ghg_s1 = base_year_ghg_s2 = base_year_ghg_s3 = reduction_ambition = 0
                start_year = end_year = None

                # Iterate through all rows for this company
                for _, target_row in company_targets.iterrows():
                    # Update values if they exist and are not null
                    for col in ['base_year_ghg_s1', 'base_year_ghg_s2', 'base_year_ghg_s3', 'reduction_ambition', 'start_year', 'end_year']:
                        if col in target_row and pd.notna(target_row[col]):
                            locals()[col] = target_row[col]

                # Ensure we have valid years
                if start_year is None or end_year is None:
                    st.warning(f"Missing start or end year for company: {company_name}. Using default values.")
                    start_year = start_year or 2020
                    end_year = end_year or 2030

                # Calculate emission reductions and temperature scores
                emissions_df = calculate_emissions_reduction(
                    base_year_ghg_s1,
                    base_year_ghg_s2,
                    base_year_ghg_s3,
                    reduction_ambition,
                    start_year,
                    end_year
                )
                emissions_df['Company Name'] = company_name
                emission_results.append(emissions_df)

            if not emission_results:
                st.error("No emission results calculated. Please check your input data.")
                return

            # Combine emission reduction results
            final_results = pd.concat(emission_results, ignore_index=True)

            if final_results.empty:
                st.error("No results to display. Please check your input data and ensure targets are available for companies.")
                return

            st.header("Results Visualization")
            st.subheader("Initial Emissions Breakdown")
            total_initial_s1 = final_results.groupby('Company Name')['S1 Emissions'].first().sum()
            total_initial_s2 = final_results.groupby('Company Name')['S2 Emissions'].first().sum()
            total_initial_s3 = final_results.groupby('Company Name')['S3 Emissions'].first().sum()

            pie_data = pd.DataFrame({
                'Scope': ['Scope 1', 'Scope 2', 'Scope 3'],
                'Emissions': [total_initial_s1, total_initial_s2, total_initial_s3]
            })
            fig_pie = px.pie(pie_data, values='Emissions', names='Scope',
                             title='Initial Emissions Breakdown',
                             labels={'Emissions': 'Total Emissions'},
                             hole=0.3,
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_pie)

            # Display results table
            st.header("Emission Reductions and Temperature Scores")
            st.dataframe(final_results)

            # Visualizations
            st.header("Results Visualization")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Temperature Scores Heatmap")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(scores_df, annot=True, fmt=".2f", cmap="coolwarm",
                            cbar_kws={'label': 'Temperature Score'})
                plt.title("Portfolio Temperature Scores")
                plt.xlabel("Horizon")
                plt.ylabel("Scopes")
                st.pyplot(fig)

            with col2:
                st.subheader("3D Interactive Plot")
                # Prepare data for 3D plot
                viz_df = pd.melt(scores_df.reset_index(),
                                 id_vars=['index'],
                                 value_vars=scores_df.columns)
                viz_df.columns = ['Scope', 'Time Period', 'Temperature Score']

                fig = go.Figure(data=[go.Scatter3d(
                    x=viz_df['Scope'],
                    y=viz_df['Time Period'],
                    z=viz_df['Temperature Score'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=viz_df['Temperature Score'],
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    hoverinfo='text',
                    text=viz_df.apply(lambda
                                          row: f"Scope: {row['Scope']}<br>Time Period: {row['Time Period']}<br>Temperature Score: {row['Temperature Score']:.2f}",
                                      axis=1)
                )])

                fig.update_layout(scene=dict(
                    xaxis_title='Scope',
                    yaxis_title='Time Period',
                    zaxis_title='Temperature Score'),
                    margin=dict(r=0, b=0, l=0, t=0))

                st.plotly_chart(fig)

                # Emissions Trend
            st.subheader("Emissions Trend")
            fig = px.line(final_results, x='Year',
                          y=['S1 Emissions', 'S2 Emissions', 'S3 Emissions', 'Total Emissions'],
                          color='Company Name', title='Emissions Trend Over Time')
            st.plotly_chart(fig)

            # Temperature Score Trend
            st.subheader("Temperature Score Trend")
            fig = px.line(final_results, x='Year', y='Temperature Score', color='Company Name',
                          title='Temperature Score Trend Over Time')
            st.plotly_chart(fig)

            # Download link for results
            csv = final_results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="temperature_score_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input files and try again.")

        if __name__ == "__main__":
            create_app()