"""
Positional Inverted Index and Boolean/Phrase Query Search Engine for FinGuard.
Features:
1. Positional Index: Postings with exact token word offsets within documents.
2. Phrase Matching: Verifies consecutive token offsets within the same document.
3. Boolean Query Parser: Evaluates complex boolean queries with:
   - Operators: AND, OR, NOT
   - Explicit Operator Precedence: NOT > AND > OR
   - Parentheses grouping: (A OR B) AND C
   - Quoted phrases: "shareholder capitalism"
   - Robust syntax error handling and validation
4. Persistent serialization and versioning.
"""

import os
import json
import re
from typing import Dict, List, Set, Tuple, Any

INDEX_VERSION = "2.0-positional"

def tokenize_with_positions(text: str) -> List[Tuple[str, int]]:
    """Extracts lowercase tokens with their 0-indexed word position."""
    if not text:
        return []
    # Tokenize preserving financial tokens
    tokens = re.findall(r'\b[\$₹€£]?\w+(?:-\w+)*%?\b', text.lower())
    return [(tok, pos) for pos, tok in enumerate(tokens)]

class PositionalInvertedIndex:
    def __init__(self, corpus: List[Dict[str, Any]] = None):
        self.version = INDEX_VERSION
        # term -> {doc_id: [pos0, pos1, ...]}
        self.index: Dict[str, Dict[int, List[int]]] = {}
        self.doc_count = 0
        self.all_doc_ids: Set[int] = set()
        self.doc_metadata: Dict[int, Dict[str, Any]] = {}
        
        if corpus:
            self.build_index(corpus)

    def build_index(self, corpus: List[Dict[str, Any]]):
        """Constructs positional postings for all documents."""
        self.index.clear()
        self.all_doc_ids.clear()
        self.doc_metadata.clear()
        self.doc_count = len(corpus)
        
        for doc in corpus:
            doc_id = doc["doc_id"]
            text = doc["text"]
            self.all_doc_ids.add(doc_id)
            self.doc_metadata[doc_id] = {
                "source_urls": doc.get("source_urls", []),
                "parent_claim_ids": doc.get("parent_claim_ids", [])
            }
            
            token_positions = tokenize_with_positions(text)
            for tok, pos in token_positions:
                if tok not in self.index:
                    self.index[tok] = {}
                if doc_id not in self.index[tok]:
                    self.index[tok][doc_id] = []
                self.index[tok][doc_id].append(pos)

    def get_doc_freq(self, term: str) -> int:
        return len(self.index.get(term.lower(), {}))

    def get_term_freq(self, term: str, doc_id: int) -> int:
        return len(self.index.get(term.lower(), {}).get(doc_id, []))

    def search_term(self, term: str) -> Set[int]:
        """Returns all doc_ids containing term."""
        term_clean = term.lower().strip()
        return set(self.index.get(term_clean, {}).keys())

    def search_phrase(self, phrase: str) -> Set[int]:
        """
        Positional phrase verification.
        Finds documents where tokens occur consecutively: pos(t_{i+1}) == pos(t_i) + 1.
        """
        tokens = [t.lower() for t in re.findall(r'\b[\$₹€£]?\w+(?:-\w+)*%?\b', phrase)]
        if not tokens:
            return set()
        if len(tokens) == 1:
            return self.search_term(tokens[0])
            
        first_term = tokens[0]
        if first_term not in self.index:
            return set()
            
        # Candidates must contain all terms in phrase
        candidates = set(self.index[first_term].keys())
        for t in tokens[1:]:
            if t not in self.index:
                return set()
            candidates &= set(self.index[t].keys())
            
        matching_docs = set()
        for doc_id in candidates:
            # Check consecutive positions within this document
            first_positions = self.index[first_term][doc_id]
            for start_pos in first_positions:
                matched = True
                for offset, next_term in enumerate(tokens[1:], start=1):
                    target_pos = start_pos + offset
                    if target_pos not in self.index[next_term][doc_id]:
                        matched = False
                        break
                if matched:
                    matching_docs.add(doc_id)
                    break
        return matching_docs

    def save(self, filepath: str):
        """Persists index to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "version": self.version,
            "doc_count": self.doc_count,
            "all_doc_ids": list(self.all_doc_ids),
            "doc_metadata": {str(k): v for k, v in self.doc_metadata.items()},
            "index": {k: {str(did): pos for did, pos in postings.items()} for k, postings in self.index.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Loads index from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.version = data.get("version", INDEX_VERSION)
        self.doc_count = data.get("doc_count", 0)
        self.all_doc_ids = set(data.get("all_doc_ids", []))
        self.doc_metadata = {int(k): v for k, v in data.get("doc_metadata", {}).items()}
        self.index = {k: {int(did): pos for did, pos in postings.items()} for k, postings in data.get("index", {}).items()}


# =====================================================================
# Boolean Query Parser with Explicit Precedence (NOT > AND > OR)
# =====================================================================

class QuerySyntaxError(ValueError):
    """Raised when boolean query syntax is invalid."""
    pass

class BooleanQueryParser:
    """
    Parses boolean queries supporting:
    - Operators: AND, OR, NOT
    - Quoted phrases: "..."
    - Parentheses: (...)
    - Precedence: NOT (3) > AND (2) > OR (1)
    """
    def __init__(self, index: PositionalInvertedIndex):
        self.index = index

    def tokenize_query(self, query: str) -> List[Tuple[str, str]]:
        """
        Tokenizes query into (token_type, value) tuples.
        Types: 'PHRASE', 'OP', 'LPAREN', 'RPAREN', 'TERM'
        """
        tokens = []
        i = 0
        n = len(query)
        while i < n:
            c = query[i]
            if c.isspace():
                i += 1
                continue
            if c == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif c == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            elif c in ('"', "'"):
                # Quoted phrase
                quote_char = c
                i += 1
                start = i
                while i < n and query[i] != quote_char:
                    i += 1
                if i >= n:
                    raise QuerySyntaxError(f"Unclosed quotation mark starting at position {start-1}")
                phrase_content = query[start:i].strip()
                tokens.append(('PHRASE', phrase_content))
                i += 1
            else:
                # Word or operator
                start = i
                while i < n and not query[i].isspace() and query[i] not in ('(', ')', '"', "'"):
                    i += 1
                word = query[start:i]
                word_upper = word.upper()
                if word_upper in ('AND', 'OR', 'NOT'):
                    tokens.append(('OP', word_upper))
                else:
                    tokens.append(('TERM', word))
        return tokens

    def parse_to_postfix(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Converts infix query tokens to Reverse Polish Notation (postfix) using Shunting-Yard.
        Precedence: NOT (3) > AND (2) > OR (1)
        """
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        output = []
        op_stack = []
        
        # Check parentheses balance
        open_count = sum(1 for t, v in tokens if t == 'LPAREN')
        close_count = sum(1 for t, v in tokens if t == 'RPAREN')
        if open_count != close_count:
            raise QuerySyntaxError(f"Unbalanced parentheses: {open_count} '(' vs {close_count} ')'")
            
        for tok_type, val in tokens:
            if tok_type in ('TERM', 'PHRASE'):
                output.append((tok_type, val))
            elif tok_type == 'OP':
                while (op_stack and op_stack[-1][0] == 'OP' and 
                       precedence.get(op_stack[-1][1], 0) >= precedence.get(val, 0)):
                    output.append(op_stack.pop())
                op_stack.append((tok_type, val))
            elif tok_type == 'LPAREN':
                op_stack.append((tok_type, val))
            elif tok_type == 'RPAREN':
                while op_stack and op_stack[-1][0] != 'LPAREN':
                    output.append(op_stack.pop())
                if not op_stack:
                    raise QuerySyntaxError("Mismatched closing parenthesis.")
                op_stack.pop() # Remove LPAREN
                
        while op_stack:
            top = op_stack.pop()
            if top[0] in ('LPAREN', 'RPAREN'):
                raise QuerySyntaxError("Unbalanced parentheses detected.")
            output.append(top)
            
        return output

    def evaluate(self, query: str) -> Set[int]:
        """Evaluates boolean query against index and returns matching doc_ids."""
        if not query or not query.strip():
            return set()
            
        tokens = self.tokenize_query(query)
        if not tokens:
            return set()
            
        postfix = self.parse_to_postfix(tokens)
        eval_stack = []
        
        for tok_type, val in postfix:
            if tok_type == 'TERM':
                docs = self.index.search_term(val)
                eval_stack.append(docs)
            elif tok_type == 'PHRASE':
                docs = self.index.search_phrase(val)
                eval_stack.append(docs)
            elif tok_type == 'OP':
                if val == 'NOT':
                    if not eval_stack:
                        raise QuerySyntaxError("Unary NOT missing argument.")
                    operand = eval_stack.pop()
                    eval_stack.append(self.index.all_doc_ids - operand)
                elif val == 'AND':
                    if len(eval_stack) < 2:
                        raise QuerySyntaxError("Binary AND missing operands.")
                    right = eval_stack.pop()
                    left = eval_stack.pop()
                    eval_stack.append(left & right)
                elif val == 'OR':
                    if len(eval_stack) < 2:
                        raise QuerySyntaxError("Binary OR missing operands.")
                    right = eval_stack.pop()
                    left = eval_stack.pop()
                    eval_stack.append(left | right)
                    
        if len(eval_stack) != 1:
            raise QuerySyntaxError(f"Malformed query evaluation stack (remaining items: {len(eval_stack)})")
            
        return eval_stack[0]

if __name__ == "__main__":
    sample_corpus = [
        {"doc_id": 1, "text": "Joe Biden called for an end to shareholder capitalism during a speech."},
        {"doc_id": 2, "text": "Amazon paid zero federal corporate income tax despite record high profits."},
        {"doc_id": 3, "text": "The national debt reached twenty trillion dollars in record time."},
        {"doc_id": 4, "text": "Shareholder capitalism prioritizes corporate profit over workers and community."}
    ]
    idx = PositionalInvertedIndex(sample_corpus)
    parser = BooleanQueryParser(idx)
    
    print("Testing phrase search:")
    print('  "shareholder capitalism":', idx.search_phrase("shareholder capitalism"))
    print('  "corporate profit":', idx.search_phrase("corporate profit"))
    print('  "capitalism shareholder" (reversed):', idx.search_phrase("capitalism shareholder"))
    
    print("\nTesting Boolean queries:")
    print('  "shareholder capitalism" AND Biden:', parser.evaluate('"shareholder capitalism" AND Biden'))
    print('  Amazon OR (debt AND NOT Biden):', parser.evaluate('Amazon OR (debt AND NOT Biden)'))
    print('  NOT "shareholder capitalism":', len(parser.evaluate('NOT "shareholder capitalism"')))
