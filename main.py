import streamlit as st
import os
import random
import urllib.parse
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from tavily import TavilyClient


# 1. Load Secrets (Local .env support)
load_dotenv()

# ==========================================
# 2. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="AI News Agency", page_icon="📰", layout="wide")

# Custom CSS to make it look professional
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #0077B5; /* LinkedIn Blue */
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📰 The AI News Agency")
st.markdown("### Powered by Multi-Agent AI Team")

# ==========================================
# 3. TOOLS & HELPER FUNCTIONS
# ==========================================

# Custom Search Tool (Extreme Diet Version) for free version
class SearchTool(BaseTool):
    name: str = "Search"
    description: str = "Useful for search-based queries. Use this to find current news."

    def _run(self, query: str) -> str:
        try:
            # 1. Fetch Key
            api_key = os.environ.get("TAVILY_API_KEY") or st.secrets["TAVILY_API_KEY"]
            tavily = TavilyClient(api_key=api_key)
            
            # 2. Search - DRASTIC REDUCTION
            # We only ask for 1 result now to save token usage
            response = tavily.search(query=query, search_depth="basic", max_results=1)
            
            results = []
            for item in response['results']:
                # Take only the first 300 characters (approx 60-70 words)
                snippet = item['content'][:300]
                results.append(f"SOURCE: {item['url']}\nCONTENT: {snippet}...")
            
            # 3. CRITICAL: Pause for 5 seconds to let the API limit reset
            time.sleep(5)
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error: {str(e)}"

def generate_image_url(prompt):
    # 1. Clean the prompt (remove newlines and quotes)
    cleaned_prompt = prompt.strip().replace("\n", " ").replace('"', '').replace("'", "")
    
    # 2. URL Encode (turns "AI Robot" into "AI%20Robot" safely)
    encoded_prompt = urllib.parse.quote(cleaned_prompt)
    
    # 3. Add a random seed to ensure a new image every time
    seed = random.randint(1, 99999)
    
    # 4. Use the direct image endpoint
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?nologo=true&seed={seed}&width=1024&height=576"

# ==========================================
# 4. SIDEBAR (About Only - No Keys)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2583/2583486.png", width=100)
    st.header("About the Team")
    st.markdown("""
    **Researcher Agent:** Scours the web for real-time news using Tavily.
    
    **Writer Agent:** Drafts viral LinkedIn posts using Llama-3.
    
    **Designer Agent:** Generates cover images using Pollinations AI.
    """)
    st.divider()
    st.caption("Built by Prince | AIML Student")

# ==========================================
# 5. MAIN APP LOGIC
# ==========================================

topic = st.text_input("What topic should we cover today?", placeholder="e.g. 'Future of humanoid robots'")

if st.button("🚀 Launch News Crew"):
    
    # Check for keys in Environment
    if not os.environ.get("GROQ_API_KEY") and "GROQ_API_KEY" not in st.secrets:
        st.error("❌ API Keys not found! Please check your .env file or Streamlit Secrets.")
        st.stop()
        
    groq_key = os.environ.get("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

    # Initialize Brain
    llm = LLM(model="groq/llama-3.1-8b-instant", api_key=groq_key)
    search_tool = SearchTool()

    # --- AGENTS ---
    researcher = Agent(
        role='Senior Tech Reporter',
        goal='Uncover latest news about {topic}',
        backstory="You are a veteran journalist. You only care about facts, dates, and numbers.",
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    writer = Agent(
        role='LinkedIn Influencer',
        goal='Write a viral post about {topic}',
        backstory="You write engaging, professional LinkedIn posts. You never sound like a robot.",
        verbose=True,
        llm=llm
    )

    # --- TASKS ---
    task_research = Task(
        description="Search for latest trends in {topic}. Find 3 specific new facts.",
        expected_output="A list of 3 key facts with sources.",
        agent=researcher
    )

    task_write = Task(
        description="Write a LinkedIn post based on the research. Start with a hook. Max 150 words.",
        expected_output="The final LinkedIn post text.",
        agent=writer
    )

    # --- EXECUTION ---
    crew = Crew(agents=[researcher, writer], tasks=[task_research, task_write], verbose=True)

    with st.status("🤖 AI Team is working...", expanded=True) as status:
        st.write("🕵️‍♂️ Researcher is searching the web...")
        result = crew.kickoff(inputs={'topic': topic})
        final_post = str(result)
        status.update(label="✅ Content Generated!", state="complete", expanded=False)

    # --- DISPLAY ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Draft Post")
        st.markdown(final_post)
        st.text_area("Copy for LinkedIn", value=final_post, height=300)

    with col2:
        # --- BONUS: IMAGE GENERATION ---
        st.divider()
        st.subheader("🎨 Generated Image")
        
        with st.spinner("Designing image..."):
            # We explicitly tell the AI to ONLY return the visual description
            image_prompt_request = f"""
            Based on this post: '{final_post[:300]}...'
            
            Describe a high-quality, futuristic tech image to go with it.
            Examples: "A glowing blue brain chip", "A robot shaking hands with a human".
            CRITICAL: Return ONLY the description. No "Here is the prompt". Max 10 words.
            """
            
            # Call the LLM
            image_description = llm.call(
                [{"role": "user", "content": image_prompt_request}]
            )
            
            # Generate and Display
            image_url = generate_image_url(image_description)
            
            st.image(image_url, caption=f"Prompt: {image_description}")
            
            # Debugging: If image breaks, they can click this link to see why
            st.markdown(f"[🔗 Click here if image doesn't load]({image_url})")