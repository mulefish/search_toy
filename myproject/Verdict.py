GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def verdict(expected, actual, msg):
    passed = expected == actual
    color = GREEN if passed else YELLOW
    status = "PASS" if passed else "FAIL"
    print(f"{color}{status}: {msg}{RESET}")
    return passed


if __name__ == "__main__":
    verdict(1, 1, "simple and ought to PASS")
    verdict([1,2,3],[1,2,3], "matching collection and ought to PASS")
    verdict([1,2,'a'],[1,2,'a'], "matching collection and ought to PASS")
    verdict([1,2,'a'],[1,2,'b'], "collection type match but values do not and ought to FAIL")
    verdict([1,2,'a'],"boo", "type mismatch and ought to FAIL")
    verdict("a", "b", "simple and ought to FAIL")
