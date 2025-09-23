#!/usr/bin/env python3
"""
Memory Generation Script for Ava's Backstory
Generates connecting memories using OpenAI API based on Pillar Memories
"""

import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Add the backend app directory to Python path for imports
backend_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(backend_root))

# Load environment variables from .env file
load_dotenv(backend_root / ".env")

def load_pillar_memories() -> str:
    """Load the pillar memories content"""
    pillar_file = Path(__file__).parent / "pillar-memories.md"
    if not pillar_file.exists():
        raise FileNotFoundError(f"Pillar memories file not found: {pillar_file}")
    
    return pillar_file.read_text()

def load_prompt_template() -> str:
    """Load the LLM prompt template"""
    prompt_file = Path(__file__).parent / "memory-generation-prompt.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_file}")
    
    return prompt_file.read_text()

def setup_openai_client():
    """Initialize OpenAI client with API key using modern best practices"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Make sure .env file is properly loaded.")
    
    return OpenAI(api_key=api_key)

def generate_memories_for_period(client: OpenAI, period_prompt: str) -> str:
    """Generate memories for a specific time period using latest OpenAI model"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # Using latest 2025 model - balanced for intelligence, speed, and cost
            messages=[
                {
                    "role": "system",
                    "content": period_prompt
                },
                {
                    "role": "user", 
                    "content": "Generate the connecting memories for this time period following the specified format and guidelines."
                }
            ],
            temperature=0.7,  # Creative writing with consistency as per 2025 best practices
            max_tokens=1500,  # Reasonable limit for memory generation
            verbosity="medium"  # New 2025 parameter for response length control
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating memories: {e}")
        return f"Error generating memories: {str(e)}"

def main():
    """Main generation workflow"""
    print("🧠 Generating Ava's Connecting Memories...")
    print("=" * 50)
    
    try:
        # Load base content
        pillar_memories = load_pillar_memories()
        prompt_template = load_prompt_template()
        
        # Setup OpenAI client
        client = setup_openai_client()
        
        # Define time periods for memory generation
        time_periods = [
            {
                "name": "Ages 13-14: Coping with Mother's Loss",
                "description": "Between Mother's Loss and Father's Absence deepening - early grief processing"
            },
            {
                "name": "Ages 14-15: Father's Emotional Absence",
                "description": "During Father's emotional absence period - learning self-reliance"
            },
            {
                "name": "Ages 15-16: Building Independence",
                "description": "Between self-reliance development and ballet focus - academic growth"
            },
            {
                "name": "Ages 16-17: Ballet Peak and Injury",
                "description": "During ballet peak and injury period - identity through art and loss"
            },
            {
                "name": "Ages 17-18: After Ballet Loss",
                "description": "Between ballet loss and father's return - identity crisis and recovery"
            },
            {
                "name": "Ages 18-19: Renewed Family Bond",
                "description": "Between father's return and first love - rebuilding trust"
            },
            {
                "name": "Ages 19-21: University and Creative Discovery", 
                "description": "During university and creative discovery period - finding purpose"
            },
            {
                "name": "Ages 21-Present: Career Transition",
                "description": "Post-university transition into career/adult life - applying lessons learned"
            }
        ]
        
        # Generate memories for each period
        all_generated_memories = []
        
        for period in time_periods:
            print(f"\\n🎯 Generating memories for: {period['name']}")
            print(f"   Focus: {period['description']}")
            
            # Create period-specific prompt
            period_prompt = f"""
{prompt_template}

## CURRENT GENERATION FOCUS:
**Time Period:** {period['name']}
**Focus:** {period['description']}

## PILLAR MEMORIES CONTEXT:
{pillar_memories}

Generate 4-5 connecting memories for this specific time period that bridge to the surrounding Pillar Memories.
            """
            
            # Generate memories
            memories = generate_memories_for_period(client, period_prompt)
            
            # Store results
            period_result = f"# {period['name']}\\n\\n{memories}\\n\\n---\\n"
            all_generated_memories.append(period_result)
            
            print(f"   ✅ Generated memories for {period['name']}")
        
        # Save all generated memories to file
        output_file = Path(__file__).parent / "generated-connecting-memories.md"
        
        header = f"""# Generated Connecting Memories for Ava

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** GPT-4 (Premium LLM)
**Purpose:** Bridge Pillar Memories with smaller, character-building moments

---

"""
        
        with open(output_file, 'w') as f:
            f.write(header)
            f.write("\\n\\n".join(all_generated_memories))
        
        print(f"\\n🎉 Generation Complete!")
        print(f"📁 Generated memories saved to: {output_file}")
        print(f"📊 Generated {len(time_periods)} time periods with connecting memories")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()