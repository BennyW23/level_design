#!/usr/bin/env python3

import bisect
import itertools
import operator

n = 4

def powerset_ignore_empty(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

sets = list(powerset_ignore_empty(range(1,n + 1)))
print(sets)
print('num sets: ' + str(len(sets)))

results = set()
for duration_set in sets:
    options = []
    for pref_duration in range(1, n+1):
        # find highest <= x and lowest >= x
        # use lowest / highest if not found
        try:
            lower = find_le(duration_set, pref_duration)
        except ValueError:
            lower = duration_set[0]
        try:
            higher = find_ge(duration_set, pref_duration)
        except ValueError:
            higher = duration_set[-1]

        # if they are same, either both are x or one set is empty
        if lower == higher:
            options.append([lower])
        else:
            options.append([lower, higher])

    # find all the combinations
    products = itertools.product(*options)
    for combo in products:
        if combo not in results:
            #print(combo)
            results.add(combo)

print()
sort_key = list(range(n))
print(sorted(results, key=operator.itemgetter(*sort_key)))
print('num configs: ' + str(len(results)))
