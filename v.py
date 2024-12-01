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
        st.image("https://github.git comitcom/VerdureClimateLtd/verdure_temp_score_tool/blob/main/verd.png", width='NONE')
        st.markdown("<h2 style='display:inline; font-family:Arial;'>Verdure Climate</h2>", unsafe_allow_html=True)

        #st.markdown("---")
       # st.subheader("Companies")
       # company_dropdown = st.selectbox("Select a company:", ["Company A", "Company B", "Company C"])

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
                    company_id=index,
                    investment_value=row['investment_value'],
                    company_isin=row['company_isin']
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
                company_id = index

                # Fetch all rows for the company_id from target_data
                company_targets = target_data[target_data['company_id'] == company_id]

                if not company_targets.empty:
                    # Initialize variables with default values of 0
                    base_year_ghg_s1 = 0
                    base_year_ghg_s2 = 0
                    base_year_ghg_s3 = 0
                    reduction_ambition = 0
                    start_year = None
                    end_year = None

                    # Iterate through all rows for this company
                    for _, target_row in company_targets.iterrows():
                        # Update values if they exist and are not null
                        if 'base_year_ghg_s1' in target_row and pd.notna(target_row['base_year_ghg_s1']):
                            base_year_ghg_s1 = target_row['base_year_ghg_s1']
                        if 'base_year_ghg_s2' in target_row and pd.notna(target_row['base_year_ghg_s2']):
                            base_year_ghg_s2 = target_row['base_year_ghg_s2']
                        if 'base_year_ghg_s3' in target_row and pd.notna(target_row['base_year_ghg_s3']):
                            base_year_ghg_s3 = target_row['base_year_ghg_s3']
                        if 'reduction_ambition' in target_row and pd.notna(target_row['reduction_ambition']):
                            reduction_ambition = target_row['reduction_ambition']
                        if 'start_year' in target_row and pd.notna(target_row['start_year']):
                            start_year = target_row['start_year']
                        if 'end_year' in target_row and pd.notna(target_row['end_year']):
                            end_year = target_row['end_year']

                    # Ensure we have valid years
                    if start_year is None or end_year is None:
                        # Use default values if years are missing
                        start_year = 2020  # or any default start year
                        end_year = 2030  # or any default end year

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

            # Combine emission reduction results
            final_results = pd.concat(emission_results, ignore_index=True)
            #st.header("Emission Reductions and Temperature Scores")
            #st.dataframe(final_results)

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
                        size=10,
                        color=viz_df['Temperature Score'],
                        colorscale='Viridis',
                        showscale=True
                    )
                )])

                fig.update_layout(
                    scene=dict(
                        xaxis_title='Scope',
                        yaxis_title='Time Period',
                        zaxis_title='Temperature Score'
                    ),
                    width=700,
                    height=650
                )

                st.plotly_chart(fig)

            # New visualization: Temperature Score Trends
            st.subheader("Temperature Score Trends")
            fig = go.Figure()

            for company in final_results['Company Name'].unique():
                company_data = final_results[final_results['Company Name'] == company]
                fig.add_trace(go.Scatter(
                    x=company_data['Year'],
                    y=company_data['Temperature Score'],
                    mode='lines+markers',
                    name=company
                ))

            fig.update_layout(
                title='Temperature Score Trends by Company',
                xaxis_title='Year',
                yaxis_title='Temperature Score (°C)',
                yaxis_range=[1.5, 3.5],  # Adjust as needed
                legend_title='Company'
            )

            st.plotly_chart(fig)



            # Download results
            st.header("Download Results")
            csv = final_results.to_csv(index=False)
            st.download_button(
                label="Download emission reductions and temperature scores as CSV",
                data=csv,
                file_name="emission_reductions_and_temperature_scores.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    create_app()