# verdure_temp_score_tool
A temperature scoring tool which gives temperature scores to organizations and shows a predicted carbon reduction over time aligning to temperature pathways in line with SBTi standards and the Paris Agreement.

Verdure Climate Portfolio Analysis Tool
Welcome to the Verdure Climate Portfolio Analysis Tool! This Streamlit app is designed to help organizations and individuals assess, visualize, and manage the carbon emission profiles of companies in their investment portfolios. By leveraging Science-Based Targets initiative (SBTi) tools and other advanced data analysis techniques, this app provides actionable insights to support sustainable and climate-friendly investment decisions.

Features
Company Portfolio Analysis
Upload your portfolio file and analyze the climate performance of companies listed under the company_name column.

Emission Reduction Trend Visualization
Interactive graphs that show the emission reduction trends over time.

Temperature Scores
Calculate temperature alignment scores for companies based on SBTi methodologies.

Interactive Dropdown
Use a dropdown to select specific companies in your portfolio and view their detailed emission data.

User-Friendly Interface
Intuitive and accessible design, including explanations for non-technical users.

Customizable Options
Choose aggregation methods and timeframes for deeper portfolio analysis.

Export Results
Download reports and data for further use.

Installation
To run the app locally, follow these steps:

Clone the Repository

bash
Copy code
git clone https://github.com/<your-repo-url>/verdure-climate-app.git
cd verdure-climate-app
Install Dependencies
Make sure you have Python 3.8+ installed. Then, install the required libraries:

bash
Copy code
pip install -r requirements.txt
Run the App
Start the Streamlit server:

bash
Copy code
streamlit run app.py
Access the App
Open the provided local URL in your browser (e.g., http://localhost:8501).

How to Use
Upload Portfolio Data
Upload a .csv file containing your portfolio. The file should have a company_name column listing the companies in your portfolio.

Select a Company
Use the dropdown menu to select a company and view its temperature scores, emission reduction trends, and other key metrics.

View Aggregated Results
Explore overall portfolio trends, aggregated temperature scores, and emissions summaries.

Download Reports
Generate and download reports for further reference or sharing with stakeholders.

Technologies Used
Python: Core programming language for data manipulation and analysis.
Streamlit: Framework for building the app's interactive web interface.
Pandas: Data manipulation and processing.
Matplotlib/Seaborn: Visualization libraries for static graphs.
Plotly: For creating interactive charts and dashboards.
SBTi Tools: Science-Based Targets initiative tools for climate analysis.
NumPy: For numerical operations.
Folder Structure
bash
Copy code
verdure-climate-app/
│
├── app.py                  # Main Streamlit application script
├── requirements.txt        # Python dependencies
├── data/                   # Placeholder for example portfolio data files
├── assets/                 # Contains images, logos, and other media assets
└── README.md               # Documentation file
Contributing
We welcome contributions to enhance the app. To contribute:

Fork the repository.
Create a new branch for your feature.
Submit a pull request for review.
Support
For questions or support, please contact Verdure Climate at:
📧 admin@verdureclimate.com


