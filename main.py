import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from urllib.error import URLError
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

# Set page config and title
st.set_page_config(
    page_title="Wage Gap at the University of Utah",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': '# This app analyzes wage gaps at the University of Utah'
    }
)

# Add title and description
st.title("Wage Gap at the University of Utah")

try:
    # Load and display data
    dfu = pd.read_csv("UofUPayroll2018.csv")

    dfu2 = dfu[dfu['amount'] <300000]
    dfu2= dfu2[dfu2['amount'] >0]
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=dfu2, x='amount', hue='GENDER', common_norm=False, ax=ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000):,}'))
    plt.title('Compensation (KDE)')
    plt.ylabel('Density')
    plt.xlabel('Compensation (Thousands)')
    plt.legend(['Male','Female'])
    
    # Add a container for the chart
    with st.container():
        st.subheader("Overall Compensation")
        
        # Display the chart
        st.pyplot(fig, use_container_width=True)
        st.markdown("""
        <h4>This chart shows the distribution of compensation for all employees at the University of Utah.
        Note the higher density of female compensation at the lower end of the distribution.</h4>
        """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    dfuap = dfu[dfu['TITLE'] == 'ASSOCIATE PROFESSOR']
    sns.kdeplot(data=dfuap,x='amount', hue='GENDER', common_norm=False, ax=ax)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000):,}'))
    plt.title('Associate Professor Compensation (KDE)')
    plt.ylabel('Density')
    plt.xlabel('Compensation (Thousands)')
    plt.legend(['Male','Female'])

    with st.container():
        st.subheader("Associate Professor Compensation")
        
        # Display the chart
        st.pyplot(fig, use_container_width=True)
        st.markdown("""
        <h4>This chart shows the distribution of compensation for associate professors at the University of Utah.
        This time, with the title of the employee controlled, the result is similar, making it more likely that gender plays a role in compensation.</h4>
        """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    dfuapc = dfu[dfu['TITLE'] == 'ASSISTANT PROFESSOR (CLINICAL)']
    sns.kdeplot(data=dfuapc,x='amount', hue='GENDER', common_norm=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000):,}'))
    plt.title('Assistant Professor (Clinical) Compensation (KDE)')
    plt.ylabel('Density')
    plt.xlabel('Compensation (Thousands)')
    plt.legend(['Male','Female'])

    with st.container():
        st.subheader("Assistant Professor (Clinical) Compensation")
        
        # Display the chart
        st.pyplot(fig, use_container_width=True)
        st.markdown("""
        <h4>This chart shows the distribution of compensation for assistant professors (clinical) at the University of Utah.
        There are two clear peaks in the distribution, but the female peaks are higher on the lower end of the distribution.</h4>
        """, unsafe_allow_html=True)

    dfu_train = dfu[['TITLE','org1','amount','GENDER']]
    y=dfu_train.pop('GENDER')
    dfu_train_encoded = pd.get_dummies(dfu_train, drop_first=True)
    scaler = MinMaxScaler()
    dfu_train_encoded['amount'] = scaler.fit_transform(dfu_train_encoded[['amount']])
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(dfu_train_encoded, y)
    feature_importance_df = pd.DataFrame(zip(dfu_train_encoded.columns, np.transpose(rf_model.feature_importances_)), 
                                       columns=['features', 'importance']).sort_values('importance', ascending=False)
    
    with st.container():
        st.subheader("Feature Importance Analysis")
        st.markdown("""
        <h4>This table shows the relative importance of different features in predicting gender-based compensation patterns.
        Higher importance values indicate features that have a stronger influence on the model's predictions.</h4>
        """, unsafe_allow_html=True)
        
        # Display the feature importance table with formatting
        st.dataframe(
            feature_importance_df.style.format({'importance': '{:.4f}'}),
            use_container_width=True
        )

    scaler = MinMaxScaler()
    dfu_train_encoded = pd.get_dummies(dfu_train.drop('TITLE',axis = 1), drop_first=True)
    dfu_scaled = dfu_train_encoded.astype(int)
    dfu_scaled['amount'] = scaler.fit_transform(dfu_scaled[['amount']])
    X2 = sm.add_constant(dfu_scaled)
    est = sm.Logit(y, X2)
    est2 = est.fit()
    with st.container():
        st.subheader("Logistic Regression Model")
        st.write(est2.summary())
        st.markdown("""
        <h4>This table shows the results of the logistic regression model.
        When the 15.0342 coefficient for amount is unscaled, we get 0.000004, meaning that a one dollar increase in compensation increases the odds of the person
        being male by 0.0004%. This doesn't seem that large, but when there is a 10,000 dollar increase,
        odds for being male go up by 4%.</h4>
        """, unsafe_allow_html=True)
    

    dfu_train = dfu[['TITLE','org1','amount','GENDER']]
    y=dfu_train.pop('amount')
    dfu_train_encoded = pd.get_dummies(dfu_train, drop_first=True)
    y_alt = y.apply(lambda x: int(x/10000))
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(dfu_train_encoded, y_alt)
    feature_importance_df = pd.DataFrame(zip(dfu_train_encoded.columns, np.transpose(rf_model.feature_importances_)), columns=['features', 'importance']).sort_values('importance', ascending=False)
    with st.container():
        st.subheader("Feature Importance Analysis")
        st.markdown("""
        <h4>Like the previous analysis, this table shows the relative importance of different features in predicting gender-based compensation patterns.
        However, this time, we test the gender and other features to see if they are important in predicting the amount of compensation.</h4>
        """, unsafe_allow_html=True)
        st.dataframe(
            feature_importance_df.style.format({'importance': '{:.4f}'}),
            use_container_width=True
        )
        

except URLError as e:
    st.error(f"⚠️ This demo requires internet access. Connection error: {e.reason}")