# Documentation Directory

This directory contains documentation for custom utility modules and Python files created for my personal and professional LLM projects. Each documentation file provides detailed explanations of classes, methods, and usage examples.

## Current Documentation

### AutoDoc Module
**File**: `autoDoc.md`  
**Purpose**: Documents the automated SQL view documentation generator  
**Key Components**:  
- `MetadataLoggingHandler`: Tracks LLM operations and runtime
- `Documentation`: Pydantic model for structured documentation 
- Utility functions for view extraction and context handling

[View Documentation](./autoDoc.md)

### Document Manager 
**File**: `document_loader.md`  
**Purpose**: Documents the document loading and RAG system facade  
**Key Components**:
- Multi-format document loading (PDF, Word, Excel, PowerPoint)
- Text chunking and vector storage
- RAG-based question answering

[View Documentation](./document_loader.md)

## Adding New Documentation

When adding new documentation:
1. Create a markdown file named after the module
2. Include overview, class/function descriptions, and usage examples
3. Use clear headings and code blocks for examples
4. Update this README with the new documentation entry

## Structure Guidelines

Each documentation file should include:
- Module overview
- Installation/setup requirements  
- Class and function documentation
- Usage examples
- Notes and considerations
- Links to related modules

## Usage

```bash
# Clone the repository
git clone <repository-url>

# Navigate to documentation
cd utils/documentation

# View documentation
# On Windows
start documentation_file.md
# On Unix
open documentation_file.md