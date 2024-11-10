from itertools import chain, combinations
from collections import defaultdict
import matplotlib.pyplot as plt

# Sample dataset
transactions = [
    {"Milk", "Bread", "Butter"},
    {"Bread", "Butter"},
    {"Milk", "Bread", "Sugar"},
    {"Milk", "Sugar"},
    {"Milk", "Bread", "Butter", "Sugar"},
    {"Bread", "Sugar"},
    {"Milk", "Bread"}
]

# Function to calculate all itemsets and their support
def calculate_support(transactions):
    itemsets_support = defaultdict(int)
    total_transactions = len(transactions)
    
    # Generate all possible itemsets for each transaction and count their occurrences
    for transaction in transactions:
        for itemset in chain.from_iterable(combinations(transaction, r) for r in range(1, len(transaction) + 1)):
            itemsets_support[frozenset(itemset)] += 1
    
    # Calculate support as fraction of transactions
    for itemset in itemsets_support:
        itemsets_support[itemset] /= total_transactions
    
    return itemsets_support

# Task 1: Calculate support for all itemsets
itemsets_support = calculate_support(transactions)

# Task 2: Frequent Itemset Generation based on a minimum support threshold
min_support_threshold = 0.3  # Example threshold
frequent_itemsets = {itemset: support for itemset, support in itemsets_support.items() if support >= min_support_threshold}

# Task 3: Calculate confidence for association rules
def calculate_confidence(frequent_itemsets):
    rules_confidence = {}
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for antecedent in chain.from_iterable(combinations(itemset, r) for r in range(1, len(itemset))):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if antecedent in frequent_itemsets:
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                    rules_confidence[(antecedent, consequent)] = confidence
    return rules_confidence

rules_confidence = calculate_confidence(frequent_itemsets)

# Task 4: Calculate lift for association rules
def calculate_lift(rules_confidence, frequent_itemsets):
    rules_lift = {}
    for (antecedent, consequent), confidence in rules_confidence.items():
        if consequent in frequent_itemsets:
            lift = confidence / frequent_itemsets[consequent]
            rules_lift[(antecedent, consequent)] = lift
    return rules_lift

rules_lift = calculate_lift(rules_confidence, frequent_itemsets)

# Task 5: Visualize Frequent Itemsets and their Support Values
itemset_labels = [' '.join(itemset) for itemset in frequent_itemsets.keys()]
support_values = list(frequent_itemsets.values())

plt.figure(figsize=(10, 6))
plt.bar(itemset_labels, support_values, color='skyblue')
plt.xlabel("Frequent Itemsets")
plt.ylabel("Support")
plt.title("Frequent Itemsets and Their Support Values")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Output results
itemsets_support, frequent_itemsets, rules_confidence, rules_lift
