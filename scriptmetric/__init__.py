"""SCRIPT: a reference-free measure of behavioural scriptedness from span annotations.

    from scriptmetric import metric        # compute(), matched_ceiling(), profile_distance(), profile_loglik()
    from scriptmetric import betting       # SCRIPT-Seq, the anytime-valid sequential test

Command line:
    python -m scriptmetric.metric  score spans.csv
    python -m scriptmetric.betting test  spans.csv --alpha 0.05
"""
