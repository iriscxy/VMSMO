import requests


def eval_rouge(project_name, exp_name, run_name, decoded, reference):
    try:
        requests.post('http://172.31.20.117:5001/eval/rouge/add_task', json={
            'project_name': project_name,
            'exp_name': exp_name,
            'run_name': run_name,
            'data': {
                'decoded': decoded,
                'reference': reference
            }
        }, timeout=10)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    eval_rouge('test', 'legal_debugging', '444', [
        [
            'das dsfs sfs fds sf sd',
            'das dsfs sfs fds sf sd',
            'das dsfs sfs fds sf sd',
        ],
    [
            'das dsfs sfs fds sf sd',
            'das dsfs sfs fds sf sd',
            'das dsfs sfs fds sf sd',
        ],
    [
            'das dsfs sfs fds sf sd',
            'das dsfs sfs fds sf sd',
            'das dsfs sfs fds sf sd',
        ],
    ], [
    [
            'das dsfs sfs fds sf sd',
            'das dsds sfs fds sf sd',
            'das dsfs sfs fd sf sd'
        ],
    [
            'das dsfs sfs fds sf sd',
            'das dsfs sfs ffs sf sd',
            'das dffs sfs fds sf sd',
        ],
    [
            'das dsfs sfs fds sf sd',
            'das dsfs sfs ffs sf sd',
            'das dfs sfs dds sf sd',
        ],
    ])