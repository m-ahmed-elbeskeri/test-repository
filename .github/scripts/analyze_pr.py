import os
import json
import requests
from typing import Optional, Dict, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI  # Fixed import
from langchain.agents import initialize_agent, AgentType
from atlassian import Confluence

# ============================================================================
# CONFLUENCE TOOLKIT
# ============================================================================

class ConfluenceToolkit:
    def __init__(self, url: str, username: str, api_token: str):
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True
        )
    
    def get_confluence_spaces(self) -> List[Dict]:
        """Get all Confluence spaces"""
        try:
            spaces = self.confluence.get_all_spaces(start=0, limit=50)
            return spaces.get('results', [])
        except Exception as e:
            return [{"error": str(e)}]
    
    def search_confluence_using_cql(self, cql: str) -> List[Dict]:
        """Search Confluence using CQL"""
        try:
            results = self.confluence.cql(cql)
            return results.get('results', [])
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_confluence_page(self, page_id: str) -> Dict:
        """Get a specific Confluence page"""
        try:
            page = self.confluence.get_page_by_id(
                page_id, 
                expand='body.storage,space,version'
            )
            return page
        except Exception as e:
            return {"error": str(e)}
    
    def get_pages_in_confluence_space(self, space_key: str) -> List[Dict]:
        """Get all pages in a Confluence space"""
        try:
            pages = self.confluence.get_all_pages_from_space(
                space=space_key, 
                start=0, 
                limit=25
            )
            return pages
        except Exception as e:
            return [{"error": str(e)}]

# ============================================================================
# CONFLUENCE TOOLS USING @tool DECORATOR
# ============================================================================

# Global toolkit instance - will be set in main()
confluence_toolkit = None

@tool
def get_confluence_spaces() -> str:
    """Get all available Confluence spaces with their keys and names."""
    spaces = confluence_toolkit.get_confluence_spaces()
    simplified = [{"key": s.get("key"), "name": s.get("name"), "type": s.get("type")} 
                 for s in spaces if not s.get("error")]
    return json.dumps(simplified, indent=2)

@tool 
def search_confluence_using_cql(cql: str) -> str:
    """Search Confluence using CQL query.
    
    Examples:
    - Find pages by title: 'title ~ "API" AND space = "DEV"'
    - Find pages with content: 'text ~ "authentication" AND space = "DEV"'
    - Find recent pages: 'created >= "2024-01-01" AND space = "DEV"'
    
    Args:
        cql: The CQL query string to search Confluence
    """
    results = confluence_toolkit.search_confluence_using_cql(cql)
    simplified = []
    for r in results:
        if not r.get("error"):
            simplified.append({
                "id": r.get("id"),
                "title": r.get("title"),
                "space": r.get("space", {}).get("key"),
                "url": r.get("_links", {}).get("webui"),
                "excerpt": r.get("excerpt", "")[:200]
            })
    return json.dumps(simplified, indent=2)

@tool
def get_confluence_page(page_id: str) -> str:
    """Get detailed content of a specific Confluence page by ID.
    
    Args:
        page_id: The Confluence page ID to retrieve
    """
    page = confluence_toolkit.get_confluence_page(page_id)
    if page.get("error"):
        return json.dumps(page)
    
    simplified = {
        "id": page.get("id"),
        "title": page.get("title"),
        "space": page.get("space", {}).get("key"),
        "version": page.get("version", {}).get("number"),
        "content_preview": page.get("body", {}).get("storage", {}).get("value", "")[:500],
        "url": page.get("_links", {}).get("webui")
    }
    return json.dumps(simplified, indent=2)

@tool
def get_pages_in_confluence_space(space_key: str) -> str:
    """Get all pages in a specific Confluence space by space key.
    
    Args:
        space_key: The Confluence space key (e.g., 'DEV', 'PROD')
    """
    pages = confluence_toolkit.get_pages_in_confluence_space(space_key)
    if isinstance(pages, list) and pages and pages[0].get("error"):
        return json.dumps(pages[0])
    
    simplified = []
    for p in pages[:20]:  # Limit to first 20 pages
        simplified.append({
            "id": p.get("id"),
            "title": p.get("title"),
            "status": p.get("status"),
            "url": p.get("_links", {}).get("webui")
        })
    return json.dumps(simplified, indent=2)

# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class PRConfluenceAnalyzer:
    def __init__(self):
        global confluence_toolkit
        
        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            model=os.getenv('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet'),
            temperature=0.1,
            openai_api_key=os.getenv('OPENROUTER_API_KEY'),
            openai_api_base="https://openrouter.ai/api/v1",
            model_kwargs={
                "headers": {
                    "HTTP-Referer": "https://github.com",
                    "X-Title": "GitHub Documentation Bot"
                }
            }
        )
        
        # Initialize Confluence toolkit
        confluence_toolkit = ConfluenceToolkit(
            url=os.getenv('CONFLUENCE_URL'),
            username=os.getenv('CONFLUENCE_USERNAME'),
            api_token=os.getenv('CONFLUENCE_API_TOKEN')
        )
        
        # Tools list - these are now properly decorated functions
        self.tools = [
            get_confluence_spaces,
            search_confluence_using_cql,
            get_confluence_page,
            get_pages_in_confluence_space
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True
        )
        
        # PR details
        self.pr_number = os.getenv('PR_NUMBER')
        self.repo_name = os.getenv('REPO_NAME')
        self.pr_title = os.getenv('PR_TITLE', '')
        self.pr_body = os.getenv('PR_BODY', '')
    
    def get_pr_changes(self):
        """Get PR file changes from GitHub API"""
        url = f"https://api.github.com/repos/{self.repo_name}/pulls/{self.pr_number}/files"
        headers = {
            'Authorization': f'token {os.getenv("GITHUB_TOKEN")}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ GitHub API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error fetching PR changes: {e}")
            return []
    
    def analyze_file_changes(self, files):
        """Analyze the file changes to understand impact"""
        analysis = {
            'total_files': len(files),
            'files_by_type': {},
            'significant_changes': [],
            'documentation_indicators': []
        }
        
        for f in files:
            # File type analysis
            ext = f['filename'].split('.')[-1] if '.' in f['filename'] else 'no-ext'
            analysis['files_by_type'][ext] = analysis['files_by_type'].get(ext, 0) + 1
            
            # Significant changes
            if f.get('additions', 0) > 20 or f.get('deletions', 0) > 10:
                analysis['significant_changes'].append({
                    'file': f['filename'],
                    'status': f['status'],
                    'additions': f.get('additions', 0),
                    'deletions': f.get('deletions', 0)
                })
            
            # Documentation indicators
            filename_lower = f['filename'].lower()
            if any(indicator in filename_lower for indicator in 
                   ['api', 'endpoint', 'route', 'controller', 'service']):
                analysis['documentation_indicators'].append(f"🔗 API-related: {f['filename']}")
            elif any(indicator in filename_lower for indicator in 
                     ['config', 'env', 'setting', 'properties']):
                analysis['documentation_indicators'].append(f"⚙️ Configuration: {f['filename']}")
            elif any(indicator in filename_lower for indicator in 
                     ['model', 'schema', 'migration', 'sql']):
                analysis['documentation_indicators'].append(f"🗄️ Data model: {f['filename']}")
            elif any(indicator in filename_lower for indicator in 
                     ['test', 'spec']):
                analysis['documentation_indicators'].append(f"🧪 Testing: {f['filename']}")
        
        return analysis
    
    def run_confluence_analysis(self):
        """Run the main Confluence analysis using the LangChain agent"""
        
        # Get PR changes
        files = self.get_pr_changes()
        if not files:
            return "❌ Could not retrieve PR file changes"
        
        # Analyze files
        file_analysis = self.analyze_file_changes(files)
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
        You are a documentation expert analyzing a Pull Request to determine what Confluence documentation needs updating.

        **PR Information:**
        - Repository: {self.repo_name}
        - PR #{self.pr_number}: {self.pr_title}
        - Description: {self.pr_body[:400]}{'...' if len(self.pr_body) > 400 else ''}

        **Code Changes Analysis:**
        - Total files changed: {file_analysis['total_files']}
        - File types: {json.dumps(file_analysis['files_by_type'], indent=2)}
        - Significant changes: {json.dumps(file_analysis['significant_changes'], indent=2)}
        - Documentation indicators: {json.dumps(file_analysis['documentation_indicators'], indent=2)}

        **Your Mission:**
        1. **Explore Confluence Structure**: Use get_confluence_spaces to understand what documentation spaces exist
        2. **Find Relevant Documentation**: Use search_confluence_using_cql to find pages that might need updating based on the code changes
        3. **Examine Specific Pages**: Use get_confluence_page to look at relevant pages in detail
        4. **Identify Actions Needed**: Determine what specific updates or new pages are required

        **Search Strategy:**
        - Look for API documentation if API files changed
        - Find configuration docs if config files changed  
        - Search for feature docs related to modified components
        - Check for existing pages that mention the changed files or functionality

        **Final Output:** Provide specific, actionable Confluence recommendations in JSON format:
        ```json
        {{
            "confluence_actions": [
                {{
                    "action": "update_page",
                    "page_id": "123456",
                    "space_key": "DEV", 
                    "page_title": "API Reference",
                    "reason": "New authentication endpoints added in auth/login.py",
                    "priority": "high",
                    "specific_changes": "Add documentation for POST /auth/login and PUT /auth/refresh endpoints",
                    "existing_content_summary": "Brief summary of current page content"
                }},
                {{
                    "action": "create_page",
                    "space_key": "DEV",
                    "parent_page_id": "789012", 
                    "new_page_title": "User Authentication Guide",
                    "reason": "New auth system requires user guide",
                    "priority": "medium",
                    "content_outline": "Introduction, Setup, Usage Examples, Troubleshooting"
                }}
            ],
            "summary": "Brief summary of documentation impact and rationale",
            "total_actions": 2,
            "estimated_effort": "Medium",
            "spaces_affected": ["DEV", "PROD"]
        }}
        ```

        Start your analysis now by exploring the Confluence spaces and finding relevant documentation!
        """
        
        try:
            print("🤖 Starting LangChain agent analysis with OpenRouter...")
            result = self.agent.run(analysis_prompt)
            return result
        except Exception as e:
            return f"❌ Error during LangChain analysis: {str(e)}"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("🚀 Starting PR Confluence Analysis with OpenRouter")
    print("=" * 60)
    
    # Validate environment variables
    required_vars = ['OPENROUTER_API_KEY', 'CONFLUENCE_URL', 'CONFLUENCE_USERNAME', 
                    'CONFLUENCE_API_TOKEN', 'GITHUB_TOKEN', 'PR_NUMBER']
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    # Show model being used
    model = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet')
    print(f"🤖 Using model: {model}")
    
    # Initialize analyzer
    try:
        analyzer = PRConfluenceAnalyzer()
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    print(f"📋 Analyzing PR #{analyzer.pr_number}: {analyzer.pr_title}")
    print(f"🏢 Repository: {analyzer.repo_name}")
    
    # Run the analysis
    print("\n🔍 Running Confluence analysis...")
    result = analyzer.run_confluence_analysis()
    
    print("\n" + "="*60)
    print("📊 CONFLUENCE ANALYSIS RESULTS")
    print("="*60)
    print(result)
    
    # Save raw results
    with open('confluence_analysis_raw.txt', 'w') as f:
        f.write(result)
    
    # Try to extract and save JSON
    try:
        # Look for JSON in the result
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = result[start_idx:end_idx]
            json_data = json.loads(json_str)
            
            # Save structured JSON
            with open('confluence_actions.json', 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print("\n✅ Structured results saved to:")
            print("   📄 confluence_actions.json - Structured data")
            print("   📄 confluence_analysis_raw.txt - Full agent output")
            
            # Print summary
            if 'confluence_actions' in json_data:
                actions = json_data['confluence_actions']
                print(f"\n📈 Summary: {len(actions)} documentation actions identified")
                for i, action in enumerate(actions, 1):
                    print(f"   {i}. {action.get('action', 'unknown').title()}: {action.get('page_title', action.get('new_page_title', 'Unknown'))}")
        else:
            print("\n⚠️ Could not extract structured JSON from results")
            print("📄 Raw analysis saved to confluence_analysis_raw.txt")
            
    except json.JSONDecodeError as e:
        print(f"\n⚠️ JSON parsing error: {e}")
        print("📄 Raw analysis saved to confluence_analysis_raw.txt")
    except Exception as e:
        print(f"\n⚠️ Error processing results: {e}")
    
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
