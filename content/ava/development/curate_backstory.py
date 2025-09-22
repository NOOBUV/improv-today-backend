#!/usr/bin/env python3
"""
Backstory Curation Script using Psychological Memory Evaluation Framework
Analyzes generated memories using research-based criteria and creates final corpus
"""

import os
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Tuple

# Load environment
backend_root = Path(__file__).parent.parent.parent.parent
load_dotenv(backend_root / ".env")

def setup_openai_client():
    """Initialize OpenAI client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")
    return OpenAI(api_key=api_key)

def load_evaluation_framework() -> str:
    """Load the psychological evaluation framework"""
    framework_file = Path(__file__).parent / "memory-evaluation-framework.md"
    return framework_file.read_text()

def load_generated_memories() -> str:
    """Load all generated memories"""
    memories_file = Path(__file__).parent / "generated-connecting-memories.md"
    return memories_file.read_text()

def load_pillar_memories() -> str:
    """Load pillar memories for context"""
    pillar_file = Path(__file__).parent / "pillar-memories.md"
    return pillar_file.read_text()

def parse_memories(content: str) -> List[Dict]:
    """Parse memory text into structured format"""
    memories = []
    
    # Split by memory headers
    memory_sections = re.split(r'### Memory:', content)[1:]  # Skip first empty split
    
    for section in memory_sections:
        lines = section.strip().split('\n')
        if not lines:
            continue
            
        title = lines[0].strip()
        memory_text = section.strip()
        
        # Extract structured fields if they exist
        age_match = re.search(r'\*\*Age/Time Period:\*\* (.+)', memory_text)
        context_match = re.search(r'\*\*Context:\*\* (.+)', memory_text)
        event_match = re.search(r'\*\*Event:\*\* (.+)', memory_text)
        response_match = re.search(r'\*\*Ava\'s Response:\*\* (.+)', memory_text)
        insight_match = re.search(r'\*\*Character Insight:\*\* (.+)', memory_text)
        
        memory = {
            'title': title,
            'full_text': memory_text,
            'age': age_match.group(1) if age_match else 'Unknown',
            'context': context_match.group(1) if context_match else '',
            'event': event_match.group(1) if event_match else '',
            'response': response_match.group(1) if response_match else '',
            'insight': insight_match.group(1) if insight_match else ''
        }
        memories.append(memory)
    
    return memories

def evaluate_memory_psychologically(client: OpenAI, memory: Dict, framework: str, pillar_context: str) -> Dict:
    """Evaluate a single memory using the psychological framework"""
    
    evaluation_prompt = f"""
{framework}

## MEMORY TO EVALUATE:

**Title:** {memory['title']}
**Age:** {memory['age']}
**Context:** {memory['context']}
**Event:** {memory['event']}
**Ava's Response:** {memory['response']}
**Character Insight:** {memory['insight']}

## PILLAR MEMORIES CONTEXT (for reference):
{pillar_context}

## EVALUATION TASK:
Score this memory on each of the 10 psychological dimensions (1-5 scale).
Provide specific justification for each score based on psychological principles.

Return your evaluation in this EXACT format:
```
DIMENSION_SCORES:
1. Emotional Intensity & Valence: X/5 - [justification]
2. Survival/Threat Significance: X/5 - [justification] 
3. Identity Formation Impact: X/5 - [justification]
4. Autobiographical Reasoning Value: X/5 - [justification]
5. Memory Specificity & Vividness: X/5 - [justification]
6. Social/Relational Learning: X/5 - [justification]
7. Developmental Timing Significance: X/5 - [justification]
8. Narrative Coherence & Connection: X/5 - [justification]
9. Unconscious Processing Likelihood: X/5 - [justification]
10. Psychological Defense Relevance: X/5 - [justification]

TOTAL_SCORE: XX/50
QUALITY_TIER: [Premium/High-Quality/Moderate/Low-Quality]
PSYCHOLOGICAL_SIGNIFICANCE: [2-3 sentence summary of why this memory matters psychologically]
```
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a clinical psychologist specializing in autobiographical memory research. Evaluate memories based on established psychological principles of memory formation and retention."
                },
                {
                    "role": "user",
                    "content": evaluation_prompt
                }
            ],
            temperature=0.3,  # Lower temperature for analytical evaluation
            max_tokens=1000
        )
        
        evaluation_text = response.choices[0].message.content
        
        # Parse the scores
        scores = {}
        total_score = 0
        quality_tier = "Unknown"
        significance = ""
        
        # Extract dimension scores
        for i in range(1, 11):
            pattern = rf'{i}\. [^:]+: (\d)/5'
            match = re.search(pattern, evaluation_text)
            if match:
                score = int(match.group(1))
                scores[f'dimension_{i}'] = score
                total_score += score
        
        # Extract total score and tier
        total_match = re.search(r'TOTAL_SCORE: (\d+)/50', evaluation_text)
        if total_match:
            total_score = int(total_match.group(1))
            
        tier_match = re.search(r'QUALITY_TIER: ([^\n]+)', evaluation_text)
        if tier_match:
            quality_tier = tier_match.group(1).strip()
            
        sig_match = re.search(r'PSYCHOLOGICAL_SIGNIFICANCE: ([^\n]+(?:\n[^\n]+)*)', evaluation_text)
        if sig_match:
            significance = sig_match.group(1).strip()
        
        return {
            'evaluation_text': evaluation_text,
            'scores': scores,
            'total_score': total_score,
            'quality_tier': quality_tier,
            'psychological_significance': significance
        }
        
    except Exception as e:
        print(f"Error evaluating memory '{memory['title']}': {e}")
        return {
            'evaluation_text': f"Error: {str(e)}",
            'scores': {},
            'total_score': 0,
            'quality_tier': "Error",
            'psychological_significance': "Evaluation failed"
        }

def regenerate_weak_memory(client: OpenAI, memory: Dict, evaluation: Dict, pillar_context: str, framework: str) -> Dict:
    """Regenerate a weak memory with targeted psychological improvements"""
    
    # Analyze what dimensions scored low
    weak_dimensions = []
    eval_text = evaluation['evaluation_text']
    
    for i in range(1, 11):
        pattern = rf'{i}\. [^:]+: ([123])/5 - ([^\n]+)'  # Include 1-3 scores as improvable
        match = re.search(pattern, eval_text)
        if match:
            weak_dimensions.append({
                'dimension_num': i,
                'score': int(match.group(1)),
                'justification': match.group(2)
            })
    
    if len(weak_dimensions) < 1:  # Need at least 1 dimension scoring 1-3 to regenerate
        return None
    
    # Create improvement instructions
    improvements = []
    for dim in weak_dimensions:
        if dim['dimension_num'] == 1:
            improvements.append("ADD STRONG EMOTIONAL INTENSITY: Include vivid emotional reactions, physical sensations, strong feelings")
        elif dim['dimension_num'] == 2:
            improvements.append("CONNECT TO SURVIVAL/THREAT: Show how this relates to Ava's fears of abandonment or loss of control")
        elif dim['dimension_num'] == 3:
            improvements.append("MAKE IT IDENTITY-DEFINING: Show how this moment shaped Ava's core personality traits")
        elif dim['dimension_num'] == 4:
            improvements.append("INCREASE REASONING VALUE: Make this explain WHY Ava behaves certain ways in her adult life")
        elif dim['dimension_num'] == 5:
            improvements.append("ADD SPECIFIC SENSORY DETAILS: Include sounds, smells, textures, visual details")
        elif dim['dimension_num'] == 6:
            improvements.append("ENHANCE SOCIAL LEARNING: Show important lessons about relationships or trust")
        elif dim['dimension_num'] == 7:
            improvements.append("EMPHASIZE DEVELOPMENTAL TIMING: Connect to critical adolescent development periods")
        elif dim['dimension_num'] == 8:
            improvements.append("STRENGTHEN NARRATIVE LINKS: Connect clearly to Pillar Memories and character progression")
        elif dim['dimension_num'] == 9:
            improvements.append("MAKE IT PSYCHOLOGICALLY STICKY: Create a memory Ava would mentally revisit often")
        elif dim['dimension_num'] == 10:
            improvements.append("CONNECT TO DEFENSE MECHANISMS: Show relationship to humor, self-reliance, or guardedness")
    
    regeneration_prompt = f"""
You are regenerating a psychologically weak memory to make it more authentic and impactful.

ORIGINAL MEMORY:
{memory['full_text']}

WEAKNESS ANALYSIS:
{chr(10).join(f"- Dimension {d['dimension_num']}: {d['score']}/5 - {d['justification']}" for d in weak_dimensions)}

IMPROVEMENT INSTRUCTIONS:
{chr(10).join(f"- {imp}" for imp in improvements)}

PSYCHOLOGICAL CONTEXT:
{pillar_context}

Regenerate this memory keeping the same age/time period but making it psychologically compelling. Use format:

### Memory: [Title]
**Age/Time Period:** [Age]
**Context:** [Detailed setting]
**Event:** [Specific psychologically significant event]
**Ava's Response:** [Her reaction showing psychological processing]
**Character Insight:** [What this reveals about her development]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a clinical psychologist creating psychologically authentic character memories based on memory formation research."},
                {"role": "user", "content": regeneration_prompt}
            ],
            temperature=0.8,
            max_tokens=1200
        )
        
        regenerated_text = response.choices[0].message.content
        
        # Parse the regenerated memory
        title_match = re.search(r'### Memory: (.+)', regenerated_text)
        age_match = re.search(r'\*\*Age/Time Period:\*\* (.+)', regenerated_text)
        context_match = re.search(r'\*\*Context:\*\* (.+)', regenerated_text)
        event_match = re.search(r'\*\*Event:\*\* (.+)', regenerated_text)
        response_match = re.search(r'\*\*Ava\'s Response:\*\* (.+)', regenerated_text)
        insight_match = re.search(r'\*\*Character Insight:\*\* (.+)', regenerated_text)
        
        return {
            'title': title_match.group(1) if title_match else f"Regenerated: {memory['title']}",
            'full_text': regenerated_text,
            'age': age_match.group(1) if age_match else memory['age'],
            'context': context_match.group(1) if context_match else '',
            'event': event_match.group(1) if event_match else '',
            'response': response_match.group(1) if response_match else '',
            'insight': insight_match.group(1) if insight_match else '',
            'is_regenerated': True
        }
        
    except Exception as e:
        print(f"Error regenerating memory: {e}")
        return None

def main():
    """Main curation workflow"""
    print("🧠 Psychological Backstory Curation")
    print("=" * 50)
    
    try:
        # Load all content
        print("📚 Loading content and framework...")
        framework = load_evaluation_framework()
        generated_memories = load_generated_memories()
        pillar_memories = load_pillar_memories()
        
        # Setup OpenAI client
        client = setup_openai_client()
        
        # Parse memories
        print("🔍 Parsing memories...")
        memories = parse_memories(generated_memories)
        print(f"Found {len(memories)} memories to evaluate")
        
        # Evaluate each memory
        print("\n🎯 Evaluating memories using psychological framework...")
        evaluated_memories = []
        regenerated_count = 0
        
        for i, memory in enumerate(memories, 1):
            print(f"   [{i}/{len(memories)}] Evaluating: {memory['title'][:50]}...")
            
            evaluation = evaluate_memory_psychologically(
                client, memory, framework, pillar_memories
            )
            
            memory['evaluation'] = evaluation
            print(f"      Score: {evaluation['total_score']}/50 ({evaluation['quality_tier']})")
            
            # Check if memory needs regeneration (score < 40/50 - only Premium memories kept)
            if evaluation['total_score'] < 40:  # Only Premium memories (40-50) are kept as-is
                print(f"      🔄 Non-premium memory detected - attempting regeneration...")
                
                regenerated_memory = regenerate_weak_memory(
                    client, memory, evaluation, pillar_memories, framework
                )
                
                if regenerated_memory:
                    print(f"      ✅ Regenerated - re-evaluating...")
                    new_evaluation = evaluate_memory_psychologically(
                        client, regenerated_memory, framework, pillar_memories
                    )
                    
                    # Use regenerated version if it scores higher
                    if new_evaluation['total_score'] > evaluation['total_score']:
                        regenerated_memory['evaluation'] = new_evaluation
                        regenerated_memory['original_memory'] = memory
                        evaluated_memories.append(regenerated_memory)
                        regenerated_count += 1
                        print(f"      🎉 Improved: {evaluation['total_score']} → {new_evaluation['total_score']}/50")
                    else:
                        memory['evaluation'] = evaluation
                        evaluated_memories.append(memory)
                        print(f"      📋 Kept original: {evaluation['total_score']} vs {new_evaluation['total_score']} (no improvement)")
                else:
                    memory['evaluation'] = evaluation
                    evaluated_memories.append(memory)
                    print(f"      ⚠️  No regeneration attempted (insufficient weak dimensions)")
            else:
                memory['evaluation'] = evaluation
                evaluated_memories.append(memory)
        
        print(f"\n🔄 Regeneration Summary: {regenerated_count} memories successfully improved")
        
        # Sort by total score (highest first)
        evaluated_memories.sort(key=lambda x: x['evaluation']['total_score'], reverse=True)
        
        # Create final backstory corpus
        print("\n📋 Creating final backstory corpus...")
        create_final_corpus(evaluated_memories, pillar_memories, framework)
        
        # Generate summary statistics
        print_evaluation_summary(evaluated_memories)
        
        print("\n🎉 Backstory curation complete!")
        
    except Exception as e:
        print(f"❌ Error during curation: {e}")

def create_final_corpus(evaluated_memories: List[Dict], pillar_memories: str, framework: str):
    """Create the final backstory corpus document"""
    
    # Filter memories by quality tier
    premium_memories = [m for m in evaluated_memories if m['evaluation']['quality_tier'] == 'Premium']
    high_quality = [m for m in evaluated_memories if m['evaluation']['quality_tier'] == 'High-Quality']
    moderate = [m for m in evaluated_memories if m['evaluation']['quality_tier'] == 'Moderate']
    
    corpus_content = f"""# Ava's Backstory Corpus
## Psychologically Curated Character Foundation

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Curation Method:** Psychological Memory Evaluation Framework
**Total Memories Evaluated:** {len(evaluated_memories)}
**Memories Selected:** {len(premium_memories + high_quality)} (Premium: {len(premium_memories)}, High-Quality: {len(high_quality)})

---

## Pillar Memories (Foundational Events)

{pillar_memories}

---

## Premium Connecting Memories
*Core memories essential for understanding Ava's psychology (40-50/50 score)*

"""
    
    for memory in premium_memories:
        corpus_content += f"""
### {memory['title']}
**Score:** {memory['evaluation']['total_score']}/50
**Psychological Significance:** {memory['evaluation']['psychological_significance']}

{memory['full_text']}

**Evaluation Summary:**
```
{memory['evaluation']['evaluation_text']}
```

---
"""
    
    corpus_content += f"""
## High-Quality Supporting Memories  
*Important memories that add depth and context (30-39/50 score)*

"""
    
    for memory in high_quality:
        corpus_content += f"""
### {memory['title']}
**Score:** {memory['evaluation']['total_score']}/50

{memory['full_text']}

---
"""
    
    # Add psychological analysis summary
    corpus_content += f"""
## Psychological Analysis Summary

### Character Development Patterns Identified:
1. **Humor as Defense Mechanism** - {len([m for m in premium_memories + high_quality if 'humor' in m['evaluation']['psychological_significance'].lower() or 'wit' in m['evaluation']['psychological_significance'].lower()])} memories
2. **Self-Reliance Development** - {len([m for m in premium_memories + high_quality if 'self-reliance' in m['evaluation']['psychological_significance'].lower() or 'independent' in m['evaluation']['psychological_significance'].lower()])} memories  
3. **Trauma Processing** - {len([m for m in premium_memories + high_quality if 'trauma' in m['evaluation']['psychological_significance'].lower() or 'loss' in m['evaluation']['psychological_significance'].lower()])} memories
4. **Identity Formation** - {len([m for m in premium_memories + high_quality if 'identity' in m['evaluation']['psychological_significance'].lower()])} memories

### Memory Quality Distribution:
- **Premium (40-50/50):** {len(premium_memories)} memories
- **High-Quality (30-39/50):** {len(high_quality)} memories  
- **Moderate (20-29/50):** {len(moderate)} memories
- **Low-Quality (10-19/50):** {len([m for m in evaluated_memories if m['evaluation']['quality_tier'] == 'Low-Quality'])} memories

### Recommended Usage:
This curated corpus provides a psychologically grounded foundation for Ava's character. The memories are organized by psychological significance and can be used to inform:
- Dialogue patterns and word choice
- Emotional responses to situations
- Decision-making processes
- Relationship dynamics
- Stress responses and coping mechanisms

---

*Curation completed using evidence-based psychological principles of autobiographical memory formation and retention.*
"""
    
    # Save the corpus
    output_file = Path(__file__).parent.parent / "backstory-corpus.md"
    output_file.write_text(corpus_content)
    print(f"📁 Final corpus saved to: {output_file}")

def print_evaluation_summary(evaluated_memories: List[Dict]):
    """Print summary statistics of the evaluation"""
    
    total = len(evaluated_memories)
    premium = len([m for m in evaluated_memories if m['evaluation']['quality_tier'] == 'Premium'])
    high_quality = len([m for m in evaluated_memories if m['evaluation']['quality_tier'] == 'High-Quality'])
    moderate = len([m for m in evaluated_memories if m['evaluation']['quality_tier'] == 'Moderate'])
    low_quality = len([m for m in evaluated_memories if m['evaluation']['quality_tier'] == 'Low-Quality'])
    
    avg_score = sum(m['evaluation']['total_score'] for m in evaluated_memories) / total if total > 0 else 0
    
    print(f"\n📊 Evaluation Summary:")
    print(f"   Total Memories: {total}")
    print(f"   Average Score: {avg_score:.1f}/50")
    print(f"   Premium (40-50): {premium} ({premium/total*100:.1f}%)")
    print(f"   High-Quality (30-39): {high_quality} ({high_quality/total*100:.1f}%)")
    print(f"   Moderate (20-29): {moderate} ({moderate/total*100:.1f}%)")
    print(f"   Low-Quality (10-19): {low_quality} ({low_quality/total*100:.1f}%)")

if __name__ == "__main__":
    main()