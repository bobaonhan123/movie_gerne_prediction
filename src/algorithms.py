class StringComparison:
    def __init__(self, a: str, b: str):
        self.a = a
        self.b = b

    def lcs_length(self) -> int:
        x, y = self.a, self.b
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if x[i] == y[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[m][n]

    def similarity(self) -> float:
        if not self.a or not self.b:
            return 0.0
        lcs = self.lcs_length()
        return lcs / min(len(self.a), len(self.b))