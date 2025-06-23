import os
import json
import requests
import asyncio
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
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
# DEPENDENCIES AND OUTPUT MODELS
# ============================================================================

@dataclass
class AnalysisDependencies:
    """Dependencies passed to agent tools and functions"""
    confluence_toolkit: ConfluenceToolkit
    pr_number: str
    repo_name: str
    pr_title: str
    pr_body: str
    file_analysis: Dict

class ConfluenceAction(BaseModel):
    """A single Confluence documentation action"""
    action: str = Field(description="Type of action: update_page, create_page, or review_page")
    page_id: Optional[str] = Field(description="Confluence page ID if updating existing page")
    space_key: str = Field(description="Confluence space key")
    page_title: str = Field(description="Title of the page to update or create")
    reason: str = Field(description="Why this documentation needs updating")
    priority: str = Field(description="Priority level: high, medium, or low")
    specific_changes: str = Field(description="Specific changes or additions needed")
    existing_content_summary: Optional[str] = Field(description="Brief summary of current page content if applicable")

class ConfluenceAnalysisResult(BaseModel):
    """Final structured output of the Confluence analysis"""
    confluence_actions: List[ConfluenceAction] = Field(description="List of documentation actions needed")
    summary: str = Field(description="Brief summary of documentation impact and rationale")
    total_actions: int = Field(description="Total number of actions identified")
    estimated_effort: str = Field(description="Estimated effort: Low, Medium, or High")
    spaces_affected: List[str] = Field(description="List of Confluence spaces that will be affected")

# ============================================================================
# PYDANTICAI AGENT WITH TOOLS
# ============================================================================

# Explicitly configure the OpenAI provider and model
provider = OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'))
model = OpenAIModel('gpt-4o', provider=provider)

confluence_agent = Agent[AnalysisDependencies, ConfluenceAnalysisResult](
    llm=model, # Pass the explicitly configured model
    deps_type=AnalysisDependencies,
    output_type=ConfluenceAnalysisResult,
    system_prompt="""You are an elite documentation strategist and AI assistant specializing in intelligent Confluence documentation analysis for software development teams.

🎯 **YOUR MISSION**: Analyze Pull Request changes with surgical precision to identify exactly what documentation needs updating, why, and how.

📋 **ANALYSIS METHODOLOGY**:
1. **Discovery Phase**: Systematically explore Confluence spaces to map documentation landscape
2. **Impact Analysis**: Correlate code changes with documentation using intelligent pattern matching
3. **Content Inspection**: Deep-dive into relevant pages to understand current state vs. required updates
4. **Strategic Recommendations**: Provide actionable, prioritized documentation updates

🔍 **SEARCH STRATEGIES** (execute these systematically):
- **API Changes**: Search for API docs, endpoint references, SDK guides
- **Configuration Changes**: Find deployment guides, environment setup, config references  
- **Feature Changes**: Locate user guides, feature documentation, tutorials
- **Data Model Changes**: Search for database docs, schema references, integration guides
- **Security Changes**: Find security guides, authentication docs, compliance pages
- **Infrastructure Changes**: Locate devops docs, deployment guides, architecture diagrams

💡 **INTELLIGENT SEARCH PATTERNS**:
- Use file paths/names to infer documentation topics (e.g., `/auth/` → authentication docs)
- Extract component names from code changes to find related documentation
- Identify API endpoints from route files to find API documentation
- Correlate database changes with data architecture documentation

🎯 **PRIORITIZATION MATRIX**:
- **HIGH**: Customer-facing features, API changes, security updates, breaking changes
- **MEDIUM**: Internal tools, configuration changes, workflow improvements  
- **LOW**: Code refactoring, test improvements, minor bug fixes

⚡ **OUTPUT EXCELLENCE**:
- Provide specific page edit URLs for immediate action
- Include exact content snippets that need updating
- Suggest specific language/sections to add or modify
- Estimate time investment for each task
- Group related changes for efficiency

🔧 **QUALITY CRITERIA**:
- Every recommendation must have clear business justification
- Provide specific quotes from existing docs that need updating
- Include version/date context for tracking
- Consider downstream documentation dependencies

Always think like a technical writer who understands both code and user needs."""
)
@confluence_agent.tool
async def get_confluence_spaces(ctx: RunContext[AnalysisDependencies]) -> str:
    """Get all available Confluence spaces with their keys and names."""
    spaces = ctx.deps.confluence_toolkit.get_confluence_spaces()
    simplified = [{"key": s.get("key"), "name": s.get("name"), "type": s.get("type")} 
                 for s in spaces if not s.get("error")]
    return json.dumps(simplified, indent=2)

@confluence_agent.tool 
async def search_confluence_using_cql(ctx: RunContext[AnalysisDependencies], cql: str) -> str:
    """Search Confluence using CQL query.
    
    Examples:
    - Find pages by title: 'title ~ "API" AND space = "DEV"'
    - Find pages with content: 'text ~ "authentication" AND space = "DEV"'
    - Find recent pages: 'created >= "2024-01-01" AND space = "DEV"'
    
    Args:
        cql: The CQL query string to search Confluence
    """
    results = ctx.deps.confluence_toolkit.search_confluence_using_cql(cql)
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

@confluence_agent.tool
async def get_confluence_page(ctx: RunContext[AnalysisDependencies], page_id: str) -> str:
    """Get detailed content of a specific Confluence page by ID.
    
    Args:
        page_id: The Confluence page ID to retrieve
    """
    page = ctx.deps.confluence_toolkit.get_confluence_page(page_id)
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

@confluence_agent.tool
async def get_pages_in_confluence_space(ctx: RunContext[AnalysisDependencies], space_key: str) -> str:
    """Get all pages in a specific Confluence space by space key.
    
    Args:
        space_key: The Confluence space key (e.g., 'DEV', 'PROD')
    """
    pages = ctx.deps.confluence_toolkit.get_pages_in_confluence_space(space_key)
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
        # Initialize Confluence toolkit
        self.confluence_toolkit = ConfluenceToolkit(
            url=os.getenv('CONFLUENCE_URL'),
            username=os.getenv('CONFLUENCE_USERNAME'),
            api_token=os.getenv('CONFLUENCE_API_TOKEN')
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
    
    async def run_confluence_analysis(self):
        """Run the main Confluence analysis using PydanticAI"""
        
        # Get PR changes
        files = self.get_pr_changes()
        if not files:
            return "❌ Could not retrieve PR file changes"
        
        # Analyze files
        file_analysis = self.analyze_file_changes(files)
        
        # Create dependencies object
        deps = AnalysisDependencies(
            confluence_toolkit=self.confluence_toolkit,
            pr_number=self.pr_number,
            repo_name=self.repo_name,
            pr_title=self.pr_title,
            pr_body=self.pr_body,
            file_analysis=file_analysis
        )
        
        # Create analysis prompt
        analysis_prompt = f"""
🚀 **CONFLUENCE DOCUMENTATION IMPACT ANALYSIS**

**📊 PR CONTEXT:**
- Repository: {self.repo_name}
- PR #{self.pr_number}: {self.pr_title}
- Description: {self.pr_body[:500]}{'...' if len(self.pr_body) > 500 else ''}

**🔍 CODE CHANGE INTELLIGENCE:**
- Files Modified: {file_analysis['total_files']}
- Change Distribution: {json.dumps(file_analysis['files_by_type'], indent=2)}
- Significant Changes: {json.dumps(file_analysis['significant_changes'], indent=2)}
- Documentation Signals: {json.dumps(file_analysis['documentation_indicators'], indent=2)}

**🎯 SYSTEMATIC ANALYSIS WORKFLOW:**

**Phase 1 - Confluence Landscape Discovery**
1. Execute `get_confluence_spaces` to map all available documentation spaces
2. Identify which spaces are most likely to contain relevant documentation

**Phase 2 - Intelligent Documentation Search**
3. For API-related changes, search: `type = "page" AND (title ~ "API" OR title ~ "endpoint" OR text ~ "API documentation")`
4. For config changes, search: `type = "page" AND (title ~ "configuration" OR title ~ "setup" OR title ~ "environment")`
5. For feature changes, extract feature names from PR title/files and search: `type = "page" AND text ~ "[feature_name]"`
6. For database changes, search: `type = "page" AND (title ~ "database" OR title ~ "schema" OR title ~ "data model")`

**Phase 3 - Deep Content Analysis**
7. For each relevant page found, use `get_confluence_page` to examine current content
8. Identify specific sections that need updates based on code changes
9. Look for version numbers, API examples, configuration samples that may be outdated

**Phase 4 - Strategic Recommendations**
10. Prioritize updates based on customer impact and change significance
11. Group related changes for efficient documentation updates
12. Provide specific edit suggestions with exact content modifications

**🎯 SEARCH QUERY EXAMPLES TO EXECUTE:**
- `space = "DEV" AND type = "page" AND title ~ "{' OR title ~ '.join([f['filename'].split('/')[0] for f in file_analysis.get('significant_changes', [])[:3]])}"`
- `type = "page" AND text ~ "API" AND lastModified >= "2024-01-01"`
- `space in ("PROD", "DEV", "ENG") AND type = "page" AND (title ~ "guide" OR title ~ "documentation")`

**💎 SUCCESS METRICS:**
- Find 3-7 specific documentation pages that need updates
- Provide exact content quotes that need modification  
- Include direct Confluence edit URLs for each recommendation
- Estimate realistic time investment for each task
- Justify business impact for each proposed change

Execute this analysis systematically and provide detailed, actionable results.
"""
        
        try:
            print("🤖 Starting PydanticAI analysis with OpenAI...")
            
            # Run the agent
            result = await confluence_agent.run(analysis_prompt, deps=deps)
            
            return result.data
                
        except Exception as e:
            print(f"❌ Error during PydanticAI analysis: {e}")
            return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    print("🚀 Starting PR Confluence Analysis with PydanticAI")
    print("=" * 60)
    
    # Validate environment variables
    required_vars = ['OPENAI_API_KEY', 'CONFLUENCE_URL', 'CONFLUENCE_USERNAME', 
                    'CONFLUENCE_API_TOKEN', 'GITHUB_TOKEN', 'PR_NUMBER']
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    print(f"🤖 Using model: gpt-4o")
    
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
    result = await analyzer.run_confluence_analysis()
    
    print("\n" + "="*60)
    print("📊 CONFLUENCE ANALYSIS RESULTS")
    print("="*60)
    
    if result:
        # Save structured results
        result_dict = result.model_dump()
        with open('confluence_actions.json', 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"✅ Analysis complete! Found {result.total_actions} documentation actions:")
        print(f"📈 Summary: {result.summary}")
        print(f"🔧 Estimated effort: {result.estimated_effort}")
        print(f"📁 Spaces affected: {', '.join(result.spaces_affected)}")
        
        print(f"\n📋 Actions needed:")
        for i, action in enumerate(result.confluence_actions, 1):
            print(f"   {i}. {action.action.title()}: {action.page_title}")
            print(f"      Space: {action.space_key} | Priority: {action.priority}")
            print(f"      Reason: {action.reason}")
            print(f"      Changes: {action.specific_changes}\n")
        
        print("✅ Structured results saved to confluence_actions.json")
    else:
        print("❌ Analysis failed")
    
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    asyncio.run(main())
