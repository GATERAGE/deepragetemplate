# main.py (c) 2025 RAGE

from src.generator import DeepSeekRAGE
from src.utils import setup_logging
import argparse

def main():
    parser = argparse.ArgumentParser(description='DeepSeek RAGE System')
    parser.add_argument('--knowledge-dir', default='./knowledge',
                       help='Knowledge base directory')
    parser.add_argument('--query', required=True,
                       help='Query to process')
    parser.add_argument('--output-format', choices=['json', 'markdown'],
                       default='json', help='Output format')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Initialize system
    rage = DeepSeekRAGE(knowledge_dir=args.knowledge_dir)
    
    # Load knowledge base
    rage.load_knowledge_base()
    
    # Process query
    result = rage.generate_response(args.query)
    
    # Save response
    rage.save_response(args.query, result, format=args.output_format)
    
    # Print response
    print("\nResponse:", result["response"])
    print("\nSources used:")
    for source in result["sources"]:
        print(f"- {source.get('filename', 'Unknown source')}")

if __name__ == "__main__":
    main()
