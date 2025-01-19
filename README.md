This dataset offers a comprehensive insight into the economic trajectories of nine major economies from the onset of the COVID-19 pandemic through the beginning of 2024. It encompasses crucial economic indicators and financial market data, covering aspects such as manufacturing and services performance, consumer sentiment, monetary policies, inflation rates, unemployment rates, and overall economic output. Additionally, it includes price data for each economy, with values compared against the dollar for clarity. With data spanning this period, the dataset provides valuable insights for analysts, researchers, and stakeholders into the impact of the pandemic and other significant events on these economies, facilitating an assessment of their resilience, challenges, and opportunities.

Countries included : Australia / Canada / China / Europe / Japan / New Zealand / Switzerland / United Kingdom / United States

Time Series Model Overview:
The model leverages this rich dataset for forecasting and analyzing trends across multiple economies. A SARIMAX (Seasonal Autoregressive Integrated Moving Average + exogenous variables) model is implemented to account for seasonal variations and trends specific to each country and indicator.

Key Model Features:
Multi-Country Analysis: Models specific trends for each country while enabling cross-comparative analysis.
Indicator-Specific Insights: Provides granular forecasts for individual economic indicators.
Temporal Trends: Captures both short-term fluctuations and long-term economic trajectories.

Cloud Integration with Gradio and HTML:
To enhance accessibility and usability, the model is deployed using Gradio and an HTML interface, enabling seamless interaction via cloud-based platforms. The user workflow is as follows:
Upload Data: Users can upload new datasets or parameters through the HTML page.
Interactive Visualization: Outputs, including time series forecasts, are displayed dynamically in the cloud.
Insights Generation: Forecasts and visualizations can be downloaded or integrated into decision-making workflows.
This integrated approach ensures that policymakers, analysts, and researchers can easily interact with the model, making it a powerful tool for understanding economic dynamics and crafting informed responses to challenges and opportunities.
