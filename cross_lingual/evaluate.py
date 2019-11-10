import os
import numpy as np
from sklearn import metrics
from pre_process.tag_schema import iobes_iob


def evaluate_ner(parameters, preds, dataset, id_to_tag, id_to_word):
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    preds = [p.cpu().numpy() for p in preds]
    pred_labels = []
    real_labels = []
    for d, p in zip(dataset, preds):
        assert len(d['words']) == len(p)
        p_tags = [id_to_tag[y_pred] for y_pred in p]
        r_tags = [id_to_tag[y_real] for y_real in d['tags']]
        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(p, d['tags'])):
            new_line = " ".join(
                [id_to_word[d['words'][i]]] + [r_tags[i], p_tags[i]]
            )
            predictions.append(new_line)
            count[y_real, y_pred] += 1
            pred_labels.append(y_pred)
            real_labels.append(y_real)
        predictions.append("")

    # Write predictions to disk and run CoNLL script externally
    eval_id = np.random.randint(1000000, 2000000)
    eval_temp = "tmp"
    output_path = os.path.join(eval_temp, "eval.%i.output" % eval_id)
    # scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)
    with open(output_path, 'w') as f:
        f.write("\n".join(predictions))

    # os.system("%s < %s > %s" % (eval_script, output_path, scores_path))
    print(output_path)

    pred_res = metrics.classification_report(real_labels, pred_labels)  # , target_names=list(id_to_tag.keys())[:17])
    print('Prediction results: \n{}'.format(pred_res))
    acc = float(count.trace() / max(1, count.sum()))
    f1 = metrics.f1_score(real_labels, pred_labels, average='weighted')

    return float(f1), float(acc), "\n".join(predictions)

    # CoNLL evaluation results
    '''
    if os.path.exists(scores_path):
        eval_lines = [l.rstrip() for l in open(scores_path)]
        for line in eval_lines:
            print(line)

        # Confusion matrix with accuracy for each tag
        # print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
        #     "ID", "NE", "Total",
        #     *([id_to_tag[i] for i in range(n_tags)] + ["Percent"])
        # ))
        # for i in range(n_tags):
        #     print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
        #         str(i), id_to_tag[i], str(count[i].sum()),
        #         *([count[i][j] for j in range(n_tags)] +
        #           ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        #     ))

        # Global accuracy
        print("%i/%i (%.5f%%)" % (
            count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
        ))

        # find all float numbers in string
        acc, precision, recall, f1 = re.findall("\d+\.\d+", eval_lines[1])
        acc = float(count.trace() / max(1, count.sum()))


        # Remove temp files
        os.remove(output_path)
        os.remove(scores_path)

        return float(f1), float(acc), "\n".join(predictions)
    else:
        os.remove(output_path)
        return 0.0, 0.0, "\n".join(predictions)
    '''