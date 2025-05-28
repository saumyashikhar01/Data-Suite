def format_metrics(metrics_dict):
    return "\n".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
