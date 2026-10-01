/* Pathwise (autodiff-equivalent) forward-mode derivative of experiments/reservoir/laws_v6/laws.c::law in mode 0
 * (law FA: flood pool with capacity evacuation to the release target Qe = Qc) with an optional rule-curve flux r,
 * no withdrawal. It follows the branch each min/max/floor takes, exactly as reverse-mode autodiff through the
 * engine's piecewise-linear pool fluxes does: a capacity-limited capture day carries no kc or phi derivative.
 *
 * Parameters (columns of dQ, n x 4, row-major): d/d ln T0, d/d ln kc (Qc = kc Ibar), d/d logit phi, d/d ln z
 * (Fmax = z Ibar). Q receives the release series (identical to laws.c with the same arguments).
 * Compile: cc -O2 -shared -fPIC -o liblawgrad.so lawgrad.c
 */
void lawgrad(const double *I, const double *r, int n, double T, double Qc, double phi, double Fmax, int floor_on,
             double *Q, double *dQ)
{
    double F = 0.0, S = T * I[0];
    double dF[4] = {0, 0, 0, 0};
    double dS[4] = {T * I[0], 0, 0, 0};           /* S0 = T I0 */
    const double dT[4] = {T, 0, 0, 0};
    const double dQc[4] = {0, Qc, 0, 0};
    const double dphi[4] = {0, 0, phi * (1.0 - phi), 0};
    const double dFmax[4] = {0, 0, 0, Fmax};
    for (int t = 0; t < n; t++) {
        double c = 0.0, dc[4] = {0, 0, 0, 0};
        double ex = I[t] - Qc;
        if (ex > 0.0) {
            c = phi * ex;
            for (int k = 0; k < 4; k++) dc[k] = dphi[k] * ex - phi * dQc[k];
        }
        if (Fmax > 0.0 && c > Fmax - F) {
            if (Fmax - F > 0.0) {
                c = Fmax - F;
                for (int k = 0; k < 4; k++) dc[k] = dFmax[k] - dF[k];
            } else {
                c = 0.0;
                for (int k = 0; k < 4; k++) dc[k] = 0.0;
            }
        }
        double p = I[t] - c, dp[4];
        for (int k = 0; k < 4; k++) dp[k] = -dc[k];
        double h = Qc - p, e = 0.0, de[4] = {0, 0, 0, 0};
        if (h > 0.0) {
            if (h < F) {
                e = h;
                for (int k = 0; k < 4; k++) de[k] = dQc[k] - dp[k];
            } else {
                e = F;
                for (int k = 0; k < 4; k++) de[k] = dF[k];
            }
        }
        F += c - e;
        for (int k = 0; k < 4; k++) dF[k] += dc[k] - de[k];
        double x = p + e - (r ? r[t] : 0.0), dx[4];
        for (int k = 0; k < 4; k++) dx[k] = dp[k] + de[k];
        double avail = S + x, dav[4];
        for (int k = 0; k < 4; k++) dav[k] = dS[k] + dx[k];
        double q = avail / (T + 1.0), dq[4];
        for (int k = 0; k < 4; k++) dq[k] = dav[k] / (T + 1.0) - avail / ((T + 1.0) * (T + 1.0)) * dT[k];
        if (floor_on && q < 0.0) {
            q = 0.0;
            for (int k = 0; k < 4; k++) dq[k] = 0.0;
        }
        S = avail - q;
        for (int k = 0; k < 4; k++) dS[k] = dav[k] - dq[k];
        Q[t] = q;
        for (int k = 0; k < 4; k++) dQ[4 * t + k] = dq[k];
    }
}
