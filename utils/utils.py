__author__ = 'mateuszopala'


def generate_submission(ids, hazards, path):
    with open(path, 'w') as f:
        f.write('Id,Hazard\n')
        for hazard_id, hazard in zip(ids, hazards):
            f.write('%d, %f\n' % (hazard_id, hazard))