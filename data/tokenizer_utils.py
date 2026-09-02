from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        result = []
        max_len = max(len(k) for k in vocab) if vocab else 1

        for num in numbers:
            s = str(num)
            tokens = []
            i = 0
            n = len(s)
            while i < n:
                matched = None
                for length in range(min(max_len, n-i), 0, -1):
                    candidate = s[i:i+length]
                    if candidate in vocab:
                        matched = candidate
                        break
                if matched is None:
                    matched = s[i]
                tokens.append(matched)
                i+= len(matched)
            result.append(tokens)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        total = 0
        max_len = max(len(k) for k in vocab) if vocab else 1

        i = 0
        n = len(text)
        while i < n:
            matched = None
            for length in range(min(max_len, n - i), 0, -1):
                candidate = text[i:i+length]
                if candidate in vocab:
                    matched = candidate
                    break
            if matched is None:
                matched = text[i]
            total += 1
            i += len(matched)

        return total

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        words = text.split()
        if not words:
            return 0.0
        
        total_tokens = self.count_tokens(text,vocab)
        fertility = total_tokens/len(words)
        return round(fertility ,4)
        
