import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
import time

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    downloads = [
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('punkt', 'tokenizers/punkt'), 
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('omw-1.4', 'corpora/omw-1.4')
    ]
    
    for name, path in downloads:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except:
                pass  # Continue if download fails

# Initialize NLTK downloads
download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="NLP Text Processor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .step-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
    }
    
    .info-box {
        background: rgba(79, 172, 254, 0.1);
        border-left: 4px solid #4facfe;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(40, 167, 69, 0.1);
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 NLP Text Processor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Text Preprocessing & Analysis Pipeline</p>', unsafe_allow_html=True)

# Cleaning function
def clean_text(text):
    """Enhanced text cleaning function"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W+', ' ', text)  # Replace non-word characters with spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    return text.strip()

# Enhanced web scraping function
def scrape_webpage(url):
    """Enhanced web scraping function that works with any webpage"""
    try:
        # Add headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = 'https://' + url
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        content = []
        
        # Get title
        title = soup.find('title')
        if title:
            content.append(f"# {title.get_text().strip()}")
        else:
            content.append(f"# Scraped Content from {parsed_url.netloc}")
        
        # Extract main content
        # Try to find main content areas first
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=re.compile(r'content|main|article', re.I)) or
            soup.find('body')
        )
        
        if main_content:
            # Extract headings and paragraphs
            for tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div']):
                text = tag.get_text().strip()
                
                # Skip empty text or very short text
                if len(text) < 10:
                    continue
                    
                # Skip navigation, ads, etc.
                if any(word in text.lower() for word in ['cookie', 'advertisement', 'subscribe', 'newsletter']):
                    continue
                
                if tag.name in ['h1', 'h2', 'h3', 'h4']:
                    if text and not text.lower().startswith(('menu', 'navigation', 'footer')):
                        level = '#' * int(tag.name[1])
                        content.append(f"{level} {text}")
                elif tag.name in ['p', 'div']:
                    if text and len(text.split()) > 5:  # Only paragraphs with more than 5 words
                        content.append(text)
        
        return "\n\n".join(content) if content else "No content could be extracted from this webpage."
        
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch the webpage: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing the webpage: {str(e)}")

# Preprocessing pipeline
def preprocess_text(corpus):
    """Enhanced preprocessing pipeline"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Cleaning
    status_text.text("🧹 Cleaning text...")
    progress_bar.progress(20)
    cleaned_corpus = [clean_text(doc) for doc in corpus]
    
    # Step 2: Tokenization
    status_text.text("✂️ Tokenizing text...")
    progress_bar.progress(40)
    tokenized_corpus = [word_tokenize(doc) for doc in cleaned_corpus]
    
    # Step 3: Stop word removal
    status_text.text("🚫 Removing stop words...")
    progress_bar.progress(60)
    stop_words = set(stopwords.words('english'))
    filtered_corpus = [[word for word in doc if word not in stop_words and len(word) > 2] for doc in tokenized_corpus]
    
    # Step 4: Stemming
    status_text.text("🌱 Stemming words...")
    progress_bar.progress(80)
    stemmer = PorterStemmer()
    stemmed_corpus = [[stemmer.stem(word) for word in doc] for doc in filtered_corpus]
    
    # Step 5: Lemmatization
    status_text.text("📝 Lemmatizing words...")
    progress_bar.progress(100)
    lemmatizer = WordNetLemmatizer()
    lemmatized_corpus = [[lemmatizer.lemmatize(word) for word in doc] for doc in filtered_corpus]
    
    status_text.text("✅ Processing complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    return cleaned_corpus, tokenized_corpus, filtered_corpus, stemmed_corpus, lemmatized_corpus

# Sidebar for input selection
with st.sidebar:
    st.markdown("### 🔧 Input Configuration")
    input_type = st.selectbox(
        "Choose Input Method",
        ("📝 Enter Text", "📄 Upload File", "🌐 Scrape Webpage"),
        help="Select how you want to provide text for processing"
    )
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool provides comprehensive text preprocessing including:
    - Text cleaning & normalization
    - Tokenization
    - Stop word removal
    - Stemming & Lemmatization
    - Web scraping from any URL
    """)

# Main content area
corpus = []
scraped_content = ""

if input_type == "📝 Enter Text":
    st.markdown('<div class="info-box">💡 Enter your text below for preprocessing</div>', unsafe_allow_html=True)
    user_text = st.text_area(
        "Input Text", 
        height=200, 
        placeholder="Paste your text here for NLP preprocessing...",
        help="Enter any text you want to preprocess"
    )
    if user_text:
        corpus = [user_text]

elif input_type == "📄 Upload File":
    st.markdown('<div class="info-box">📎 Upload a text file (TXT format supported)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a text file", 
        type=["txt"],
        help="Upload a .txt file containing the text you want to process"
    )
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        corpus = [file_content]
        st.success(f"✅ File uploaded successfully! ({len(file_content)} characters)")

elif input_type == "🌐 Scrape Webpage":
    st.markdown('<div class="info-box">🔗 Enter any webpage URL to scrape its content</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        web_url = st.text_input(
            "Website URL", 
            placeholder="https://example.com or just example.com",
            help="Enter any webpage URL. The tool will extract the main content."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        scrape_btn = st.button("🔍 Scrape", type="primary")
    
    if web_url and scrape_btn:
        try:
            with st.spinner("🕷️ Scraping webpage content..."):
                scraped_content = scrape_webpage(web_url)
                corpus = [scraped_content]
            
            st.markdown('<div class="success-box">✅ Successfully scraped webpage content!</div>', unsafe_allow_html=True)
            
            # Show scraped content in an expander
            with st.expander("📄 View Scraped Content", expanded=False):
                st.text_area("Scraped Text", scraped_content, height=300, disabled=True)
                
        except Exception as e:
            st.markdown(f'<div class="warning-box">⚠️ {str(e)}</div>', unsafe_allow_html=True)

# Processing section
if corpus:
    st.markdown("---")
    
    # Show text statistics
    col1, col2, col3, col4 = st.columns(4)
    text_stats = corpus[0]
    
    with col1:
        st.metric("📊 Characters", f"{len(text_stats):,}")
    with col2:
        st.metric("📝 Words", f"{len(text_stats.split()):,}")
    with col3:
        st.metric("📄 Lines", f"{len(text_stats.splitlines()):,}")
    with col4:
        st.metric("🔤 Unique Words", f"{len(set(text_stats.lower().split())):,}")
    
    st.markdown("---")
    
    # Process button
    if st.button("🚀 Start NLP Processing", type="primary", use_container_width=True):
        st.markdown("## 🔄 Processing Pipeline")
        
        # Run preprocessing
        cleaned, tokenized, filtered, stemmed, lemmatized = preprocess_text(corpus)
        
        # Create tabs for different outputs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧹 Cleaned", "✂️ Tokenized", "🚫 Filtered", "🌱 Stemmed", "📝 Lemmatized"
        ])
        
        with tab1:
            st.markdown('<div class="step-header">1. Cleaned Text</div>', unsafe_allow_html=True)
            st.text_area("Cleaned Output", cleaned[0], height=300, disabled=True)
            
        with tab2:
            st.markdown('<div class="step-header">2. Tokenized Text</div>', unsafe_allow_html=True)
            st.write("**Sample tokens:**", tokenized[0][:50] if len(tokenized[0]) > 50 else tokenized[0])
            st.info(f"Total tokens: {len(tokenized[0]):,}")
            
        with tab3:
            st.markdown('<div class="step-header">3. Filtered Text (Stop words removed)</div>', unsafe_allow_html=True)
            st.write("**Sample tokens:**", filtered[0][:50] if len(filtered[0]) > 50 else filtered[0])
            st.info(f"Remaining tokens: {len(filtered[0]):,}")
            
        with tab4:
            st.markdown('<div class="step-header">4. Stemmed Text</div>', unsafe_allow_html=True)
            st.write("**Sample stemmed tokens:**", stemmed[0][:50] if len(stemmed[0]) > 50 else stemmed[0])
            st.info(f"Stemmed tokens: {len(stemmed[0]):,}")
            
        with tab5:
            st.markdown('<div class="step-header">5. Lemmatized Text</div>', unsafe_allow_html=True)
            st.write("**Sample lemmatized tokens:**", lemmatized[0][:50] if len(lemmatized[0]) > 50 else lemmatized[0])
            st.info(f"Lemmatized tokens: {len(lemmatized[0]):,}")
        
        # Download processed data
        st.markdown("---")
        st.markdown("### 💾 Export Processed Data")
        
        col1, col2 = st.columns(2)
        with col1:
            cleaned_text = ' '.join(lemmatized[0])
            st.download_button(
                "📥 Download Processed Text",
                cleaned_text,
                file_name="processed_text.txt",
                mime="text/plain"
            )
        with col2:
            tokens_text = '\n'.join(lemmatized[0])
            st.download_button(
                "📥 Download Tokens",
                tokens_text,
                file_name="tokens.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; padding: 1rem;'>
        🧠 NLP Text Processor | Built with Streamlit & NLTK
    </div>
    """,
    unsafe_allow_html=True
)