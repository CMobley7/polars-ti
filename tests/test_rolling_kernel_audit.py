"""The audit must report positive and negative evidence."""

from scripts.rolling_kernel_audit.audit import inspect_source


def test_audit_nested_alias_and_hidden_work():
    records = inspect_source("""
def nested(x, w):
    for i in range(len(x)):
        for j in range(w):
            pass

def sliced(x, w):
    for i in range(len(x)):
        window = x[i-w:i]
        value = np.mean(window)

def hidden(x, w):
    return np.convolve(x, w)

def recurrence(x):
    for i in range(len(x)):
        x[i] += 1
""")
    assert [row["status"] for row in records] == ["flagged", "flagged", "flagged", "clear"]
