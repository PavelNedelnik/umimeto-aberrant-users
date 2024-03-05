def submission_to_string(idx, log, item):
    # TODO maybe print side by side
    submission = log.loc[idx]
    task = item.iloc[submission['item']]

    return                                                                                          \
        f"SUBMISSION: by user: {submission['user']} of task: {submission['item']}-{task['name']}" + \
        '\n' + "-" * 50 + '\n'                                                                      \
        f"DISTANCE:\n {submission['distance']}" +                                                   \
        '\n' + "-" * 50 + '\n'                                                                      \
        f"INSTRUCTIONS:\n {task['instructions']}" +                                                 \
        '\n' + "-" * 50 + '\n'                                                                      \
        f"SOLUTION:\n {task['solution']}" +                                                         \
        '\n' + "-" * 50 + '\n'                                                                      \
        f"ANSWER:\n {submission['answer']}"