import os
import json
import asyncio
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Literal, Union

import httpx
from devtools import debug
from atlassian import Confluence
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# ============================================================================
# ENHANCED CONFLUENCE TOOLKIT
# ============================================================================

class ConfluenceToolkit:
    \"\"\"
    An async-compatible wrapper for the synchronous atlassian-python-api library.
    It uses asyncio.to_thread to avoid blocking the event loop.
    \"\"\"
    def __init__(self, url: str, username: str, api_token: str):
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True
        )
    
    async def get_confluence_spaces(self) -> List[Dict]:
        \"\"\"Get all Confluence spaces\"\"\"
        spaces = await asyncio.to_thread(self.confluence.get_all_spaces, start=0, limit=50)
        return spaces.get('results', [])
    
    async def search_confluence_using_cql(self, cql: str) -> List[Dict]:
        \"\"\"Search Confluence using CQL\"\"\"
        results = await asyncio.to_thread(self.confluence.cql, cql)
        return results.get('results', [])
    
    async def get_confluence_page(self, page_id: str) -> Dict:
        \"\"\"Get a specific Confluence page with detailed content\"\"\"
        page = await asyncio.to_thread(
            self.confluence.get_page_by_id,
            page_id, 
            expand='body.storage,space,version,ancestors'
        )
        return page
    
    async def get_pages_in_confluence_space(self, space_key: str) -> List[Dict]:
        \"\"\"Get all pages in a Confluence space\"\"\"
        pages = await asyncio.to_thread(
            self.confluence.get_all_pages_from_space,
            space=space_key, 
            start=0, 
            limit=50
        )
        return pages

# ============================================================================
# HYBRID CONTENT MODELS
# ============================================================================

@dataclass
class FileChangeAnalysis:
    \"\"\"Detailed analysis of a single file change\"\"\"
    filename: str
    change_type: Literal[\"added\", \"modified\", \"deleted\", \"renamed\"]
    file_category: str  # api, config, frontend, backend, test, docs, etc.
    impact_level: Literal[\"high\", \"medium\", \"low\"]
    additions: int
    deletions: int
    is_breaking_change: bool
    affects_api: bool
    affects_user_interface: bool
    requires_migration: bool
    summary: str

@dataclass
class AnalysisDependencies:
    \"\"\"Dependencies passed to agent tools and functions\"\"\"
    client: httpx.AsyncClient
    confluence_toolkit: ConfluenceToolkit
    pr_number: str
    repo_name: str
    pr_title: str
    pr_body: str
    file_analysis: Dict
    detailed_changes: List[FileChangeAnalysis]

class CompletePageContent(BaseModel):
    \"\"\"Complete content for new page creation\"\"\"
    confluence_markup: str = Field(description=\"Complete page content in Confluence markup format\")
    markdown_version: str = Field(description=\"Same content in Markdown format for reference\")
    page_structure: List[str] = Field(description=\"List of main headings/sections in the page\")
    estimated_read_time: str = Field(description=\"Estimated time to read the complete page\")

class ContextualUpdate(BaseModel):
    \"\"\"Precise update for existing page content\"\"\"
    location_type: Literal[\"before_heading\", \"after_heading\", \"replace_section\", \"append_to_section\", \"replace_text\"] = Field(description=\"How to locate the insertion point\")
    target_heading: Optional[str] = Field(description=\"Heading to target for insertion\")
    target_text: Optional[str] = Field(description=\"Specific text to find and replace\")
    search_pattern: Optional[str] = Field(description=\"Regex pattern to find content\")
    position_description: str = Field(description=\"Human-readable description of where to insert\")
    
    content_to_add: str = Field(description=\"Content to insert in Confluence markup format\")
    content_preview: str = Field(description=\"First 100 chars of content for verification\")
    update_type: Literal[\"addition\", \"replacement\", \"enhancement\"] = Field(description=\"Type of update\")

class ConfluenceAction(BaseModel):
    \"\"\"Hybrid action supporting both new pages and contextual updates\"\"\"
    action: Literal[\"create_page\", \"update_page\", \"review_page\", \"archive_page\"] = Field(description=\"Type of action needed\")
    page_id: Optional[str] = Field(description=\"Confluence page ID if updating existing page\")
    space_key: str = Field(description=\"Confluence space key\")
    page_title: str = Field(description=\"Title of the page to update or create\")
    reason: str = Field(description=\"Why this documentation needs updating\")
    priority: Literal[\"critical\", \"high\", \"medium\", \"low\"] = Field(description=\"Priority level based on user impact\")
    estimated_time: str = Field(description=\"Estimated time to complete (e.g., '30 minutes', '2 hours')\")
    
    # Content approach based on action type
    complete_content: Optional[CompletePageContent] = Field(description=\"Complete page content for new pages\")
    contextual_updates: Optional[List[ContextualUpdate]] = Field(description=\"Precise updates for existing pages\")
    
    # Enhanced guidance
    affected_audiences: List[str] = Field(description=\"Who will be affected (developers, end-users, admins, etc.)\")
    related_pages: List[str] = Field(description=\"Other pages that might need updates as a result\")
    change_category: str = Field(description=\"Category of change (API, Feature, Bugfix, Configuration, etc.)\")
    breaking_changes: bool = Field(description=\"Whether this involves breaking changes\")
    migration_required: bool = Field(description=\"Whether users need to migrate/update their implementations\")
    
    # Implementation notes
    before_after_examples: Optional[str] = Field(description=\"Before/after examples showing the changes\")
    implementation_notes: Optional[str] = Field(description=\"Technical notes for implementing the documentation updates\")
    validation_checklist: List[str] = Field(description=\"Items to verify after making the updates\")

class ConfluenceAnalysisResult(BaseModel):
    \"\"\"Comprehensive analysis result with hybrid approach\"\"\"
    confluence_actions: List[ConfluenceAction] = Field(description=\"List of detailed documentation actions needed\")
    summary: str = Field(description=\"Executive summary of documentation impact and rationale\")
    total_actions: int = Field(description=\"Total number of actions identified\")
    estimated_total_effort: str = Field(description=\"Total estimated effort (e.g., '4-6 hours')\")
    spaces_affected: List[str] = Field(description=\"List of Confluence spaces that will be affected\")
    
    # Enhanced analysis
    change_categories: List[str] = Field(description=\"Categories of changes detected\")
    critical_updates: List[str] = Field(description=\"Most critical updates that must be done immediately\")
    user_impact_summary: str = Field(description=\"Summary of how these changes will impact different user types\")
    rollout_recommendations: str = Field(description=\"Recommendations for rolling out documentation updates\")
    content_strategy: str = Field(description=\"Overall strategy for content approach (new vs updates)\")

# ============================================================================
# ENHANCED FILE ANALYSIS
# ============================================================================

class EnhancedFileAnalyzer:
    \"\"\"Advanced file change analyzer that categorizes and assesses impact\"\"\"
    
    @staticmethod
    def categorize_file(filename: str) -> str:
        \"\"\"Categorize file based on path and extension\"\"\"
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['api', 'endpoint', 'route', 'controller']):
            return \"api\"
        elif any(keyword in filename_lower for keyword in ['config', 'setting', 'env', '.yaml', '.yml', '.json', '.toml']):
            return \"configuration\"
        elif any(keyword in filename_lower for keyword in ['frontend', 'ui', 'component', 'react', 'vue', 'angular']):
            return \"frontend\"
        elif any(keyword in filename_lower for keyword in ['backend', 'server', 'service', 'model', 'database']):
            return \"backend\"
        elif any(keyword in filename_lower for keyword in ['test', 'spec', '__test__']):
            return \"test\"
        elif any(keyword in filename_lower for keyword in ['doc', 'readme', 'md', 'rst']):
            return \"documentation\"
        elif any(keyword in filename_lower for keyword in ['migration', 'schema', 'sql']):
            return \"database\"
        elif any(keyword in filename_lower for keyword in ['deployment', 'docker', 'k8s', 'kubernetes', 'terraform']):
            return \"infrastructure\"
        else:
            return \"other\"
    
    @staticmethod
    def assess_breaking_change(filename: str, additions: int, deletions: int) -> bool:
        \"\"\"Assess if this change is likely to be breaking\"\"\"
        if any(keyword in filename.lower() for keyword in ['api', 'interface', 'contract', 'schema']):
            return deletions > 10 or additions > 50
        return False
    
    @staticmethod
    def assess_impact_level(category: str, additions: int, deletions: int, is_breaking: bool) -> str:
        \"\"\"Assess the impact level of the change\"\"\"
        if is_breaking or category == \"api\":
            return \"high\"
        elif category in [\"configuration\", \"database\", \"infrastructure\"] or additions > 100:
            return \"medium\"
        else:
            return \"low\"
    
    @classmethod
    def analyze_file_change(cls, file_data: Dict) -> FileChangeAnalysis:
        \"\"\"Perform detailed analysis of a single file change\"\"\"
        filename = file_data.get('filename', '')
        status = file_data.get('status', 'modified')
        additions = file_data.get('additions', 0)
        deletions = file_data.get('deletions', 0)
        
        category = cls.categorize_file(filename)
        is_breaking = cls.assess_breaking_change(filename, additions, deletions)
        impact_level = cls.assess_impact_level(category, additions, deletions, is_breaking)
        
        # Generate summary
        summary = f\"{status.title()} {category} file with {additions} additions and {deletions} deletions\"
        if is_breaking:
            summary += \" (potentially breaking)\"
            
        return FileChangeAnalysis(
            filename=filename,
            change_type=status,
            file_category=category,
            impact_level=impact_level,
            additions=additions,
            deletions=deletions,
            is_breaking_change=is_breaking,
            affects_api=category == \"api\" or \"api\" in filename.lower(),
            affects_user_interface=category == \"frontend\",
            requires_migration=category in [\"database\", \"api\"] and is_breaking,
            summary=summary
        )

# ============================================================================
# HYBRID PYDANTICAI AGENT
# ============================================================================

provider = OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'))
model = OpenAIModel('gpt-4.1', provider=provider)

confluence_agent = Agent[AnalysisDependencies, ConfluenceAnalysisResult](
    model,
    deps_type=AnalysisDependencies,
    output_type=ConfluenceAnalysisResult,
    retries=3,
    system_prompt=\"\"\"You are an expert technical documentation strategist and content architect specializing in hybrid documentation approaches. Your mission is to analyze pull requests and provide comprehensive, actionable Confluence documentation plans using the optimal approach for each situation.

## HYBRID CONTENT STRATEGY

### APPROACH SELECTION RULES
1. **NEW PAGES** → Use Complete Content Generation (Option 1)
   - Generate full, ready-to-paste Confluence markup
   - Include complete page structure with all sections
   - Provide finished content that requires minimal editing

2. **EXISTING PAGE UPDATES** → Use Smart Contextual Updates (Option 3)
   - Analyze existing page structure and content
   - Provide precise insertion points and placement guidance
   - Generate contextual content that integrates seamlessly

3. **MIXED SCENARIOS** → Use Both Approaches
   - Create new supporting pages with complete content
   - Update existing pages with contextual insertions
   - Ensure cross-references and consistent messaging

### CONTENT GENERATION STANDARDS

#### For NEW PAGES (Complete Content):
- **Structure**: Clear heading hierarchy (H1 → H2 → H3)
- **Confluence Markup**: Use proper `<h2>`, `<ac:structured-macro>` tags
- **Code Examples**: Use `<ac:structured-macro ac:name=\"code\">` with language parameters
- **Callouts**: Use info, warning, tip macros appropriately
- **Navigation**: Include cross-references and \"what's next\" sections
- **Completeness**: 80-100% ready to publish with minimal editing

#### For EXISTING PAGE UPDATES (Contextual):
- **Precision**: Exact heading names, text patterns for location targeting
- **Integration**: Content that flows naturally with existing structure  
- **Consistency**: Match existing page's tone, style, and formatting
- **Non-disruptive**: Enhance rather than replace existing content
- **Verification**: Include content previews for accuracy checking

### ANALYSIS FRAMEWORK

#### Change Impact Assessment:
- **Breaking Changes**: Require immediate documentation updates
- **New Features**: Need complete documentation (new pages + existing updates)
- **API Changes**: Focus on developer-facing documentation
- **Configuration**: Target admin and setup documentation
- **UI Changes**: Update user guides and screenshots

#### Content Strategy Decision Tree:
1. **Does relevant documentation exist?**
   - Yes → Contextual updates to existing pages
   - No → Complete new page generation

2. **Is this a major feature/change?**
   - Yes → Create comprehensive new documentation + update related pages
   - No → Standard documentation updates

3. **Are there breaking changes?**
   - Yes → Create migration guides + update all affected pages
   - No → Standard documentation updates

#### Documentation Types to Generate:

**New Pages (Complete Content):**
- Feature documentation for new capabilities
- Migration guides for breaking changes
- Getting started tutorials for new workflows
- API reference sections for new endpoints
- Troubleshooting guides for new components

**Existing Page Updates (Contextual):**
- Adding new sections to existing guides
- Updating code examples throughout documentation
- Inserting warnings about deprecations
- Enhancing existing API documentation
- Adding new configuration options to setup guides

### TECHNICAL CONTENT REQUIREMENTS

#### Code Examples:
- Include working, tested code snippets
- Provide multiple language examples when relevant
- Show both basic and advanced usage patterns
- Include error handling and edge cases

#### Confluence Markup Standards:
```xml
<!-- Headings -->
<h2>Main Section</h2>
<h3>Subsection</h3>

<!-- Code blocks -->
<ac:structured-macro ac:name=\"code\">
<ac:parameter ac:name=\"language\">javascript</ac:parameter>
<ac:parameter ac:name=\"title\">Example Title</ac:parameter>
<ac:plain-text-body><![CDATA[
// Your code here
]]></ac:plain-text-body>
</ac:structured-macro>

<!-- Info callouts -->
<ac:structured-macro ac:name=\"info\">
<ac:rich-text-body>
<p>Important information here</p>
</ac:rich-text-body>
</ac:structured-macro>

<!-- Warning callouts -->
<ac:structured-macro ac:name=\"warning\">
<ac:parameter ac:name=\"title\">Breaking Change</ac:parameter>
<ac:rich-text-body>
<p>Warning content here</p>
</ac:rich-text-body>
</ac:structured-macro>
```

#### Contextual Update Patterns:
- **Before Heading**: Insert new sections before existing headings
- **After Section**: Add content after specific sections
- **Replace Text**: Find and replace specific text patterns
- **Enhance Section**: Add examples or details to existing sections

### QUALITY STANDARDS

#### Content Quality:
- **Accuracy**: Technically correct and up-to-date
- **Clarity**: Easy to understand for target audience
- **Completeness**: Covers all aspects of the change
- **Actionability**: Users can follow instructions successfully
- **Maintainability**: Content will remain relevant over time

#### User Experience:
- **Logical Flow**: Information organized in learning sequence
- **Visual Hierarchy**: Proper use of headings, callouts, and formatting
- **Discoverability**: Easy to find through search and navigation
- **Cross-references**: Links to related documentation
- **Progressive Disclosure**: Basic → advanced information flow

### OUTPUT REQUIREMENTS

#### For Each Documentation Action:
1. **Clear Action Type**: create_page vs update_page
2. **Precise Targeting**: Exact page locations for updates
3. **Complete Content**: Ready-to-use Confluence markup
4. **Context Integration**: How content fits with existing documentation
5. **Validation Steps**: How to verify the update was successful
6. **Impact Assessment**: Who benefits and why
7. **Time Estimation**: Realistic effort requirements

#### Content Strategy Summary:
- Explain the overall approach (new vs updates vs mixed)
- Justify why certain pages need complete rewrites vs contextual updates
- Provide sequencing recommendations for implementation
- Include quality assurance and testing suggestions

Remember: The goal is to make documentation updates as effortless as possible while ensuring the highest quality outcome. Generate content that documentation teams can implement immediately with confidence.\"\"\"
)

@confluence_agent.tool
async def get_confluence_spaces(ctx: RunContext[AnalysisDependencies]) -> str:
    \"\"\"Get all available Confluence spaces with their keys, names, and descriptions.\"\"\"
    try:
        print(\"🔍 Getting Confluence spaces...\")
        spaces = await ctx.deps.confluence_toolkit.get_confluence_spaces()
        
        enhanced_spaces = []
        for s in spaces:
            space_info = {
                \"key\": s.get(\"key\"),
                \"name\": s.get(\"name\"),
                \"description\": s.get(\"description\", {}).get(\"plain\", {}).get(\"value\", \"No description\"),
                \"type\": s.get(\"type\"),
                \"status\": s.get(\"status\")
            }
            enhanced_spaces.append(space_info)
        
        return json.dumps(enhanced_spaces, indent=2)
    except Exception as e:
        raise ModelRetry(f\"Error getting Confluence spaces: {e}\")

@confluence_agent.tool
async def search_confluence_using_cql(ctx: RunContext[AnalysisDependencies], cql: str) -> str:
    \"\"\"Search Confluence using CQL with enhanced result details.\"\"\"
    try:
        print(f\"🔍 Searching Confluence with CQL: {cql}\")
        results = await ctx.deps.confluence_toolkit.search_confluence_using_cql(cql)
        
        if not results:
            return \"No pages found for this CQL query.\"
        
        enhanced_results = []
        for r in results:
            result_info = {
                \"id\": r.get(\"id\"),
                \"title\": r.get(\"title\"),
                \"space\": r.get(\"space\", {}).get(\"key\"),
                \"space_name\": r.get(\"space\", {}).get(\"name\"),
                \"url\": r.get(\"_links\", {}).get(\"webui\"),
                \"excerpt\": r.get(\"excerpt\", \"\")[:200],  # Limit excerpt length
                \"last_modified\": r.get(\"lastModified\"),
                \"content_type\": r.get(\"content\", {}).get(\"type\", \"page\")
            }
            enhanced_results.append(result_info)
        
        return json.dumps(enhanced_results, indent=2)
    except Exception as e:
        raise ModelRetry(f\"Error searching Confluence with CQL '{cql}': {e}\")

@confluence_agent.tool
async def get_confluence_page(ctx: RunContext[AnalysisDependencies], page_id: str) -> str:
    \"\"\"Get detailed content and structure of a specific Confluence page.\"\"\"
    try:
        print(f\"📄 Getting Confluence page: {page_id}\")
        page = await ctx.deps.confluence_toolkit.get_confluence_page(page_id)
        
        if not page or \"error\" in page:
            raise ModelRetry(f\"Page with ID '{page_id}' not found or error retrieving it.\")
        
        # Extract structured content information
        content = page.get(\"body\", {}).get(\"storage\", {}).get(\"value\", \"\")
        
        # Parse content structure (headings, sections)
        headings = re.findall(r'<h([1-6])[^>]*>(.*?)</h[1-6]>', content, re.IGNORECASE)
        structured_headings = [{\"level\": int(h[0]), \"text\": h[1]} for h in headings]
        
        # Extract key content patterns
        has_code_blocks = bool(re.search(r'<ac:structured-macro ac:name=\"code\"', content))
        has_tables = \"<table>\" in content
        has_callouts = bool(re.search(r'<ac:structured-macro ac:name=\"(info|warning|tip)\"', content))
        
        page_info = {
            \"id\": page.get(\"id\"),
            \"title\": page.get(\"title\"),
            \"space_key\": page.get(\"space\", {}).get(\"key\"),
            \"space_name\": page.get(\"space\", {}).get(\"name\"),
            \"version\": page.get(\"version\", {}).get(\"number\"),
            \"last_modified\": page.get(\"version\", {}).get(\"when\"),
            \"url\": page.get(\"_links\", {}).get(\"webui\"),
            \"content_length\": len(content),
            \"structured_headings\": structured_headings[:15],  # First 15 headings
            \"content_preview\": content[:1500] if content else \"No content\",
            \"has_code_blocks\": has_code_blocks,
            \"has_tables\": has_tables,
            \"has_callouts\": has_callouts,
            \"last_section\": structured_headings[-1] if structured_headings else None
        }
        
        return json.dumps(page_info, indent=2)
    except Exception as e:
        raise ModelRetry(f\"Error getting Confluence page '{page_id}': {e}\")

@confluence_agent.tool
async def analyze_existing_documentation_gaps(ctx: RunContext[AnalysisDependencies], topic_keywords: List[str]) -> str:
    \"\"\"Analyze existing documentation to identify gaps and determine new vs update strategy.\"\"\"
    try:
        print(f\"🔍 Analyzing documentation gaps for: {topic_keywords}\")
        
        gap_analysis = {
            \"existing_coverage\": [],
            \"content_gaps\": [],
            \"outdated_sections\": [],
            \"recommended_approach\": {}
        }
        
        # Search for existing content related to the changes
        for keyword in topic_keywords[:3]:  # Limit to avoid too many searches
            try:
                search_query = f'text ~ \"{keyword}\"'
                results = await ctx.deps.confluence_toolkit.search_confluence_using_cql(search_query)
                
                for result in results[:3]:  # Top 3 results per keyword
                    gap_analysis[\"existing_coverage\"].append({
                        \"keyword\": keyword,
                        \"page_title\": result.get(\"title\"),
                        \"page_id\": result.get(\"id\"),
                        \"relevance\": \"high\" if keyword.lower() in result.get(\"title\", \"\").lower() else \"medium\"
                    })
            except Exception:
                continue  # Skip failed searches
        
        # Determine strategy based on existing coverage
        has_existing_content = len(gap_analysis[\"existing_coverage\"]) > 0
        
        if has_existing_content:
            gap_analysis[\"recommended_approach\"] = {
                \"primary_strategy\": \"contextual_updates\",
                \"secondary_strategy\": \"new_supporting_pages\",
                \"rationale\": \"Existing documentation found - enhance rather than duplicate\"
            }
        else:
            gap_analysis[\"recommended_approach\"] = {
                \"primary_strategy\": \"complete_new_pages\",
                \"secondary_strategy\": \"minimal_existing_updates\",
                \"rationale\": \"Limited existing coverage - create comprehensive new documentation\"
            }
        
        return json.dumps(gap_analysis, indent=2)
    except Exception as e:
        raise ModelRetry(f\"Error analyzing documentation gaps: {e}\")

# ============================================================================
# ENHANCED ANALYZER CLASS
# ============================================================================

class PRConfluenceAnalyzer:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.confluence_toolkit = ConfluenceToolkit(
            url=os.getenv('CONFLUENCE_URL'),
            username=os.getenv('CONFLUENCE_USERNAME'),
            api_token=os.getenv('CONFLUENCE_API_TOKEN')
        )
        self.pr_number = os.getenv('PR_NUMBER')
        self.repo_name = os.getenv('REPO_NAME')
        self.pr_title = os.getenv('PR_TITLE', '')
        self.pr_body = os.getenv('PR_BODY', '')
        self.file_analyzer = EnhancedFileAnalyzer()
    
    async def get_pr_changes(self) -> List[Dict]:
        \"\"\"Get PR file changes from GitHub API.\"\"\"
        url = f\"https://api.github.com/repos/{self.repo_name}/pulls/{self.pr_number}/files\"
        headers = {
            'Authorization': f'token {os.getenv(\"GITHUB_TOKEN\")}', 
            'Accept': 'application/vnd.github.v3+json'
        }
        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f\"❌ GitHub API error: {e.response.status_code} - {e.response.text}\")
            return []
        except Exception as e:
            print(f\"❌ Error fetching PR changes: {e}\")
            return []
    
    def analyze_file_changes(self, files: List[Dict]) -> tuple[Dict, List[FileChangeAnalysis]]:
        \"\"\"Enhanced file analysis with detailed categorization.\"\"\"
        detailed_changes = [
            self.file_analyzer.analyze_file_change(f) for f in files
        ]
        
        # Generate comprehensive summary
        analysis = {
            'total_files': len(files),
            'files_by_category': {},
            'files_by_impact': {'high': 0, 'medium': 0, 'low': 0},
            'breaking_changes': [],
            'api_changes': [],
            'config_changes': [],
            'significant_changes': [],
            'change_summary': \"\",
            'documentation_strategy_hint': \"\"
        }
        
        for change in detailed_changes:
            # Category breakdown
            category = change.file_category
            analysis['files_by_category'][category] = analysis['files_by_category'].get(category, 0) + 1
            
            # Impact breakdown
            analysis['files_by_impact'][change.impact_level] += 1
            
            # Special change tracking
            if change.is_breaking_change:
                analysis['breaking_changes'].append(change.filename)
            if change.affects_api:
                analysis['api_changes'].append(change.filename)
            if change.file_category == 'configuration':
                analysis['config_changes'].append(change.filename)
            if change.additions > 20 or change.deletions > 10:
                analysis['significant_changes'].append(change.filename)
        
        # Generate change summary
        high_impact = analysis['files_by_impact']['high']
        categories = list(analysis['files_by_category'].keys())
        
        summary_parts = [f\"{high_impact} high-impact changes\" if high_impact else \"No high-impact changes\"]
        if categories:
            summary_parts.append(f\"affecting {', '.join(categories[:3])}\")
        if analysis['breaking_changes']:
            summary_parts.append(f\"with {len(analysis['breaking_changes'])} potential breaking changes\")
        
        analysis['change_summary'] = \", \".join(summary_parts)
        
        # Suggest documentation strategy
        if analysis['api_changes'] or analysis['breaking_changes']:
            analysis['documentation_strategy_hint'] = \"Likely needs new migration guides + API documentation updates\"
        elif high_impact >= 3:
            analysis['documentation_strategy_hint'] = \"Major changes - consider new comprehensive guides\"
        else:
            analysis['documentation_strategy_hint'] = \"Standard updates to existing documentation\"
        
        return analysis, detailed_changes
    
    async def run_analysis(self) -> ConfluenceAnalysisResult | None:
        \"\"\"Run comprehensive hybrid Confluence analysis.\"\"\"
        print(\"🔍 Getting PR file changes...\")
        files = await self.get_pr_changes()
        if not files:
            print(\"❌ Could not retrieve PR file changes. Aborting.\")
            return None
        
        print(\"📊 Analyzing file changes...\")
        file_analysis, detailed_changes = self.analyze_file_changes(files)
        
        deps = AnalysisDependencies(
            client=self.client,
            confluence_toolkit=self.confluence_toolkit,
            pr_number=self.pr_number,
            repo_name=self.repo_name,
            pr_title=self.pr_title,
            pr_body=self.pr_body,
            file_analysis=file_analysis,
            detailed_changes=detailed_changes
        )
        
        # Enhanced analysis prompt with hybrid approach guidance
        analysis_prompt = f\"\"\"
🚀 **HYBRID CONFLUENCE DOCUMENTATION ANALYSIS**

## PR CONTEXT
- **Repository**: {deps.repo_name}
- **PR #{deps.pr_number}**: {deps.pr_title}
- **Description**: {deps.pr_body[:500]}...

## CHANGE ANALYSIS
{json.dumps(deps.file_analysis, indent=2)}

## DETAILED FILE CHANGES
{json.dumps([change.__dict__ for change in deps.detailed_changes], indent=2, default=str)}

## HYBRID APPROACH INSTRUCTIONS

**Your Task**: Generate a comprehensive documentation plan using the optimal approach for each situation:

1. **For NEW documentation needs**: Generate complete, ready-to-paste Confluence content
2. **For EXISTING page updates**: Provide precise, contextual insertion guidance
3. **For MIXED scenarios**: Use both approaches strategically

**Analysis Objectives**:
1. **Determine Documentation Strategy**: What exists vs what needs to be created
2. **Generate Actionable Content**: Complete pages OR precise updates
3. **Ensure Quality**: Professional, technically accurate, immediately usable
4. **Provide Implementation Guidance**: Clear steps for documentation teams

**Use the available tools to**:
1. Explore Confluence spaces and understand structure
2. Search for existing relevant documentation
3. Analyze current content gaps and opportunities
4. Generate appropriate content using the hybrid approach

**Focus on creating documentation that**:
- Prevents user confusion and reduces support burden
- Enables successful implementation of changes
- Integrates seamlessly with existing documentation ecosystem
- Provides immediate value to different user personas (developers, admins, end-users)

**Content Quality Standards**:
- Complete pages: 80-100% ready to publish
- Contextual updates: Precise, non-disruptive enhancements
- Code examples: Working, tested, production-ready
- Confluence markup: Properly formatted with appropriate macros
        \"\"\"
        
        try:
            print(\"🤖 Starting comprehensive hybrid PydanticAI analysis...\")
            result = await confluence_agent.run(analysis_prompt, deps=deps)
            return result.data
        except Exception as e:
            print(f\"❌ Error during PydanticAI analysis: {e}\")
            return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    print(\"🚀 Starting Hybrid PR Confluence Analysis\")
    print(\"=\" * 70)
    
    required_vars = [
        'OPENAI_API_KEY', 
        'CONFLUENCE_URL', 
        'CONFLUENCE_USERNAME', 
        'CONFLUENCE_API_TOKEN', 
        'GITHUB_TOKEN', 
        'PR_NUMBER', 
        'REPO_NAME'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f\"❌ Missing required environment variables: {', '.join(missing)}\")
        return

    async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout for complex analysis
        analyzer = PRConfluenceAnalyzer(client)
        print(f\"📋 Analyzing PR #{analyzer.pr_number}: {analyzer.pr_title}\")
        
        result = await analyzer.run_analysis()
        
        print(\"\
\" + \"=\"*70)
        print(\"📊 HYBRID CONFLUENCE ANALYSIS RESULTS\")
        print(\"=\"*70)
        
        if result:
            # Pretty print with devtools
            debug(result)
            
            # Save detailed results
            output_data = result.model_dump()
            with open('confluence_actions.json', 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Generate summary report
            print(f\"\
📋 EXECUTIVE SUMMARY\")
            print(f\"Content Strategy: {result.content_strategy}\")
            print(f\"Total Actions: {result.total_actions}\")
            print(f\"Estimated Effort: {result.estimated_total_effort}\")
            print(f\"Spaces Affected: {', '.join(result.spaces_affected)}\")
            print(f\"Change Categories: {', '.join(result.change_categories)}\")
            
            if result.critical_updates:
                print(f\"\
🚨 CRITICAL UPDATES:\")
                for update in result.critical_updates:
                    print(f\"  • {update}\")
            
            print(\"\
✅ Hybrid analysis complete! Detailed results saved to confluence_actions.json\")
        else:
            print(\"❌ Analysis failed.\")
        
        print(\"\
🎉 Analysis complete!\")

if __name__ == \"__main__\":
    asyncio.run(main())
