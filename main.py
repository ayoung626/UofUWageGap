import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.error import URLError
from matplotlib.ticker import FuncFormatter

# Set page config and title
st.set_page_config(
    page_title="Wage Gap at the University of Utah",
    page_icon="📊",
    layout="wide"
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
        st.subheader("First Visualization")
        
        # Display the chart
        st.pyplot(fig, use_container_width=True)
        
except URLError as e:
    st.error(f"⚠️ This demo requires internet access. Connection error: {e.reason}")