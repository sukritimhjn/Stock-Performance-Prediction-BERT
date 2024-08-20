# Stock Performance Prediction Based on Annual Reports

This repository contains the code and resources for the project "Stock Performance Prediction Based on Annual Reports," conducted as part of the Advanced NLP course (ANLY5800). The project aims to predict stock market performance by leveraging the textual data contained within annual reports (10-K filings) of publicly traded companies, utilizing advanced Natural Language Processing (NLP) techniques.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Objectives](#project-objectives)
3. [Significance](#significance)
4. [Literature Review](#literature-review)
5. [Expected Workflow & Outcomes](#expected-workflow--outcomes)
6. [Potential Challenges](#potential-challenges)
7. [Data Preparation](#data-preparation)
8. [Model Development](#model-development)
9. [Results & Discussion](#results--discussion)
10. [Future Work](#future-work)
11. [How to Run the Project](#how-to-run-the-project)
12. [Contributors](#contributors)

## Introduction

The primary objective of this project is to develop a robust and innovative approach for predicting stock market performance by leveraging the untapped potential of textual data contained within annual reports (10-K filings) of publicly traded companies. This project bridges the gap in financial analysis by integrating advanced NLP techniques with traditional financial metrics.

## Project Objectives

- **Utilizing Advanced NLP Techniques**: Implement state-of-the-art NLP models, particularly BERT (Bidirectional Encoder Representations from Transformers), to analyze and extract meaningful insights from the textual components of 10-K reports.
- **Enhancing Stock Performance Predictions**: Combine textual analysis insights with traditional financial metrics to improve the accuracy of stock performance predictions.
- **Data Acquisition and Processing**: Collect and process extensive datasets from annual reports and financial databases.
- **Model Development and Testing**: Build and fine-tune a predictive model to effectively process and analyze large volumes of textual and financial data.

## Significance

- **Holistic Analysis**: Analyze textual data in 10-K reports using advanced NLP techniques to introduce a more holistic approach to stock market analysis.
- **Improved Predictive Accuracy**: Integrate qualitative analysis with quantitative data for improved predictive accuracy.
- **Early Detection of Trends and Risks**: Utilize textual analysis for early detection of emerging trends, risks, and opportunities.
- **Enhanced Investor Confidence**: Provide a more comprehensive analysis to boost investor confidence.

## Literature Review

This section covers key research papers and methodologies that informed the project, including:

- **FinBERT**: A pre-trained language model tailored for financial sentiment analysis.
- **Stock Price Prediction using BERT and GAN**: Combining sentiment analysis with technical analysis for stock price prediction.
- **Stock Movement Prediction with Financial News using Contextualized Embedding from BERT**: A novel text mining method using BERT for predicting stock price movements.
- **BERT-based Financial Sentiment Index and LSTM-based Stock Return Predictability**: Integrating BERT with other methods for predicting stock returns.

## Expected Workflow & Outcomes

1. **Data Collection**: Collect annual reports and financial data.
2. **Textual Analysis**: Use BERT for deep textual analysis.
3. **Data Processing**: Clean and structure data for model input.
4. **Model Development**: Train and fine-tune a predictive model.
5. **Evaluation**: Test model accuracy and refine as needed.
6. **Actionable Insights**: Generate data-driven insights for investors and analysts.

## Potential Challenges

- **Text Length and Tokenization**: Managing the tokenization of long texts in 10-K reports.
- **Report Analysis Complexity**: Differentiating between factual and subjective content in reports.
- **Data Sufficiency and Quality**: Ensuring dataset representation across sectors.
- **Computational Limitations**: Addressing the computational challenges of large NLP models.
- **Importance of Integrating Numerical Data**: Combining numerical and textual data for comprehensive analysis.
- **Market Volatility and External Factors**: Considering external factors affecting stock performance.

## Data Preparation

- **Target Label Generation**: Label stock prices based on 7-day, 30-day, and 90-day horizons post-report release.
- **Label Balance**: Balance training data to prevent bias.
- **10-K Item Selection**: Focus on Item 1A (Risk Factors) and Item 7 (Financial Condition and Results of Operations) from 10-K reports.
- **Text Slicing**: Extract the last 512 tokens from selected report items for analysis.

## Model Development

- **Overview**: Utilize Transformer models, specifically DistilBERT and BERT.
- **Learning Rate**: Experiment with learning rates, selecting 1e-5 for optimal performance.
- **Batch Size**: Set batch size to 4 for efficient training.
- **Dropout Regularization**: Apply a 20% dropout rate to prevent overfitting.
- **Input Data Decisions**: Use a 30-day time window and focus on Item 7 for best results.
- **Final Model**: DistilBERT model selected for final predictions.

## Results & Discussion

The final model achieved a 71.5% accuracy rate in predicting stock price trends based on Item 7 of 10-K reports. While this is a significant achievement, the project also acknowledges the complexity of stock price prediction and the need for further refinement.

**Key Metrics:**
- **Accuracy**: 71.5%
- **Precision**: 75.0%
- **Recall**: 81.6%
- **F1 Score**: 78.16%

## Future Work

- **Broader Data Usage**: Expand the dataset to include reports from multiple years to enhance model generalization.
- **Enhanced Labeling**: Develop a more detailed target labeling system to differentiate between various extents of price changes.

## How to Run the Project

To run this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name/stock-performance-prediction.git
   cd stock-performance-prediction
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Jupyter Notebook**:
   If you don't have Jupyter installed, install it using pip:
   ```bash
   pip install notebook
   ```

4. **Run the Jupyter Notebook**:
   Start the Jupyter notebook server:
   ```bash
   jupyter notebook
   ```
   Open `final model.ipynb` in your browser and execute the cells to run the model.

5. **Input Data**:
   Ensure that the necessary datasets (e.g., 10-K filings, stock price data) are available in the appropriate directories or are correctly referenced in the notebook.

6. **Train the Model**:
   Execute the cells in the Jupyter notebook to preprocess the data, train the model, and evaluate the results.

7. **Analyze Results**:
   After running the notebook, review the model's performance metrics and generated predictions.

## Contributors

- Jingda Yang
- Sukriti Mahajan
- Tingsong Li
