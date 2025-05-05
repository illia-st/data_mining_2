from collections import defaultdict
from itertools import chain, combinations

class Apriori:
    def __init__(self, transactions, min_support=0.5, min_confidence=0.5):
        self.transactions = transactions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.support_cache = {}

    def find_rules(self):
        self.frequent_itemsets = self.__find_frequent_itemsets()
        return self.__find_rules()

    def __find_frequent_itemsets(self):
        n_transactions = len(self.transactions)
        candidates = defaultdict(int)

        for transaction in self.transactions:
            for item in transaction:
                candidates[frozenset([item])] += 1

        current_itemsets = {
            itemset: cnt / n_transactions
            for itemset, cnt in candidates.items()
            if cnt / n_transactions >= self.min_support
        }

        all_frequent_itemsets = dict(current_itemsets)
        k = 2

        while current_itemsets:
            candidates = defaultdict(int)
            itemsets = list(current_itemsets.keys())

            for i in range(len(itemsets)):
                for j in range(i + 1, len(itemsets)):
                    candidate = itemsets[i] | itemsets[j]

                    if len(candidate) == k and all(
                        frozenset(subset) in current_itemsets
                        for subset in combinations(candidate, k - 1)
                    ):
                        candidates[candidate] = self.__eval_support(candidate) / n_transactions

            current_itemsets = {
                itemset: support
                for itemset, support in candidates.items()
                if support >= self.min_support
            }

            all_frequent_itemsets.update(current_itemsets)
            k += 1

        return all_frequent_itemsets

    def __find_rules(self):
        rules = []

        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) > 1:
                for subset in self.__get_subsets(itemset):
                    subset_support = self.frequent_itemsets.get(subset)
                    if subset_support is None:
                        continue
                    confidence = support / subset_support
                    if confidence >= self.min_confidence:
                        rules.append((subset, itemset - subset, confidence))

        return rules

    def __get_subsets(self, itemset):
        return [
            frozenset(subset) for r in range(1, len(itemset))
            for subset in combinations(itemset, r)
        ]

    def __eval_support(self, subset):
        if subset in self.support_cache:
            return self.support_cache[subset]

        support = sum(1 for t in self.transactions if subset.issubset(t))
        self.support_cache[subset] = support
        return support

    def print_frequent_itemsets(self):
        for itemset, support in self.frequent_itemsets.items():
            print(f'{set(itemset)}, support: {support:.2f}')
    
    def print_frequent_itemsets_like_mlxtend(self):
        import pandas as pd

        df = pd.DataFrame([
            {'support': round(support, 2), 'itemsets': tuple(sorted(itemset))}
            for itemset, support in sorted(self.frequent_itemsets.items(), key=lambda x: (len(x[0]), sorted(x[0])))
        ])
        print(df)

    def print_rules(self, rules):
        for antecedent, consequent, confidence in rules:
            print(f"{set(antecedent)} => {set(consequent)}, confidence: {confidence:.2f}")
