#!/usr/bin/env python3
"""
Memory Regeneration Engine
Takes low-scoring memories and regenerates them with specific psychological instructions
to create more authentic, impactful memories that score higher on the evaluation framework
"""

import os
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict

# Load environment
backend_root = Path(__file__).parent.parent.parent.parent
load_dotenv(backend_root / ".env")

def setup_openai_client():
    """Initialize OpenAI client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")
    return OpenAI(api_key=api_key)

def analyze_weak_memory(evaluation_text: str) -> Dict:
    """Analyze why a memory scored low and identify improvement areas"""
    
    # Extract specific dimension scores that are low (1-2)
    weak_dimensions = []
    improvement_suggestions = {}
    
    dimension_names = [
        "Emotional Intensity & Valence",
        "Survival/Threat Significance", 
        "Identity Formation Impact",
        "Autobiographical Reasoning Value",
        "Memory Specificity & Vividness",
        "Social/Relational Learning",
        "Developmental Timing Significance",
        "Narrative Coherence & Connection",
        "Unconscious Processing Likelihood",
        "Psychological Defense Relevance"
    ]
    
    for i, dim_name in enumerate(dimension_names, 1):
        pattern = rf'{i}\. [^:]+: ([12])/5 - ([^\\n]+)'
        match = re.search(pattern, evaluation_text)
        if match:
            score = int(match.group(1))
            justification = match.group(2)
            weak_dimensions.append({
                'dimension': dim_name,
                'score': score,
                'justification': justification,
                'dimension_number': i
            })
    
    return {
        'weak_dimensions': weak_dimensions,
        'needs_regeneration': len(weak_dimensions) >= 3  # If 3+ dimensions score 1-2
    }

def create_regeneration_prompt(memory: Dict, weak_analysis: Dict, pillar_context: str) -> str:
    """Create targeted prompt to regenerate memory with psychological improvements"""
    
    weak_dims = weak_analysis['weak_dimensions']
    
    # Create specific improvement instructions based on weak dimensions
    improvement_instructions = []
    
    for dim in weak_dims:
        dim_num = dim['dimension_number']
        if dim_num == 1:  # Emotional Intensity & Valence
            improvement_instructions.append(
                "- INCREASE EMOTIONAL INTENSITY: Add vivid emotional reactions, physical sensations (tears, trembling, rapid heartbeat), and strong feelings that would be psychologically significant"
            )
        elif dim_num == 2:  # Survival/Threat Significance
            improvement_instructions.append(
                "- ADD SURVIVAL/THREAT ELEMENTS: Connect to Ava's core fears of abandonment, loss of control, or threats to her emotional/physical safety. Show how this memory relates to survival instincts"
            )
        elif dim_num == 3:  # Identity Formation Impact
            improvement_instructions.append(
                "- STRENGTHEN IDENTITY IMPACT: Show how this moment shaped Ava's personality - her humor, self-reliance, perfectionism, or guardedness. Make it a defining character moment"
            )
        elif dim_num == 4:  # Autobiographical Reasoning Value
            improvement_instructions.append(
                "- IMPROVE REASONING VALUE: Make this memory explain WHY Ava behaves certain ways. It should answer 'This is why I am the way I am' and connect to her major life patterns"
            )
        elif dim_num == 5:  # Memory Specificity & Vividness
            improvement_instructions.append(
                "- ADD SENSORY DETAILS: Include specific sounds, smells, textures, visual details, tastes. Make it feel like a real, vivid moment rather than a general summary"
            )
        elif dim_num == 6:  # Social/Relational Learning
            improvement_instructions.append(
                "- ENHANCE SOCIAL LEARNING: Show Ava learning important lessons about trust, relationships, social dynamics, or how to interact with others. Include relationship insights"
            )
        elif dim_num == 7:  # Developmental Timing Significance
            improvement_instructions.append(
                "- EMPHASIZE DEVELOPMENTAL TIMING: Connect to critical adolescent development - identity formation, independence, first experiences, or major transitions"
            )
        elif dim_num == 8:  # Narrative Coherence & Connection
            improvement_instructions.append(
                "- STRENGTHEN NARRATIVE CONNECTION: Clearly link to the Pillar Memories and show progression in Ava's psychological development. Create bridges between major events"
            )
        elif dim_num == 9:  # Unconscious Processing Likelihood
            improvement_instructions.append(
                "- INCREASE PSYCHOLOGICAL SIGNIFICANCE: Make this a memory Ava would mentally revisit often - either because it's painful, triumphant, or psychologically important"
            )
        elif dim_num == 10:  # Psychological Defense Relevance
            improvement_instructions.append(
                "- CONNECT TO DEFENSE MECHANISMS: Show how this memory relates to Ava's coping strategies - humor as armor, self-reliance, perfectionism, or emotional guardedness"
            )
    
    regeneration_prompt = f"""
You are a clinical psychologist and creative writer specializing in psychologically authentic character development. 

## ORIGINAL MEMORY THAT SCORED LOW:
**Title:** {memory['title']}
**Original Content:** {memory['full_text']}

## WHY IT SCORED LOW:
{chr(10).join(f"- {dim['dimension']}: {dim['score']}/5 - {dim['justification']}" for dim in weak_dims)}

## REGENERATION TASK:
Completely rewrite this memory to be more psychologically authentic and impactful. Follow these specific improvement instructions:

{chr(10).join(improvement_instructions)}

## PSYCHOLOGICAL CONTEXT (Ava's Character):
{pillar_context}

## REGENERATION REQUIREMENTS:
1. Keep the same basic age/time period as the original
2. Make it psychologically realistic - this is a memory that would actually persist in human autobiographical memory
3. Use the same format: Age/Time Period, Context, Event, Ava's Response, Character Insight
4. Ensure it connects to Ava's established psychological patterns (humor as defense, self-reliance, perfectionism, guarded intimacy)
5. Make it specific and vivid - include sensory details, emotions, and concrete actions
6. Show character growth or reinforcement of established traits

## OUTPUT FORMAT:
```
### Memory: [New Title]
**Age/Time Period:** [Age or timing]
**Context:** [Setting and circumstances with more detail]
**Event:** [What specifically happened - make it more psychologically significant]
**Ava's Response:** [Her reaction, thoughts, feelings - show psychological processing]
**Character Insight:** [What this reveals about her psychology and development]
```

Regenerate the memory now, making it psychologically compelling and authentic:
"""
    
    return regeneration_prompt

def regenerate_memory(client: OpenAI, memory: Dict, weak_analysis: Dict, pillar_context: str) -> str:
    """Regenerate a weak memory with psychological improvements"""
    
    prompt = create_regeneration_prompt(memory, weak_analysis, pillar_context)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in psychological authenticity and character development. Create memories that would realistically persist in human autobiographical memory based on psychological research."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,  # Higher creativity for regeneration
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error regenerating memory: {e}")
        return f"Regeneration failed: {str(e)}"

def main():
    """Main regeneration workflow"""
    print("🔄 Memory Regeneration Engine")
    print("=" * 50)
    
    # This would be called after the main curation script
    # For now, let's create a standalone version that can process results
    
    print("📋 This tool is designed to regenerate weak memories from the curation results.")
    print("💡 Usage: Run after curate_backstory.py to improve low-scoring memories")
    
    # Example of how to use this:
    example_usage = """
    
## USAGE EXAMPLE:

1. Run curate_backstory.py first
2. Identify memories with scores < 25/50
3. Use this script to regenerate them:

```python
# Load evaluation results
weak_memories = [memories with total_score < 25]

for memory in weak_memories:
    weak_analysis = analyze_weak_memory(memory['evaluation']['evaluation_text'])
    if weak_analysis['needs_regeneration']:
        regenerated = regenerate_memory(client, memory, weak_analysis, pillar_context)
        # Re-evaluate the regenerated memory
        # Replace in corpus if score improves
```
"""
    
    print(example_usage)

if __name__ == "__main__":
    main()