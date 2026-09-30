/* laws_v6: daily offline recursion for the flood-evacuation, withdrawal and combined laws (dt = 1 d).
 *
 * Every law is a set of fluxes on the dam row's lateral inflow followed by the per-dam linear bucket of
 * experiments/reservoir/rulecurve_offline/bucket.c (implicit Euler on storage, same-day inflow):
 *     avail = S + x_t ; q_t = avail / (T + 1) ; [floor: q_t = max(q_t, 0), deficit kept in S] ; S = avail - q_t
 * with x_t = I_t - c_t + e_t - r_t - W_t. The extra stores are explicit and carried between steps.
 *
 * Flood pool F (m3/s * d), enabled when mode >= 0: capture above a release target Qc, evacuation afterwards.
 *   capture   c_t = min(phi * max(I_t - Qc, 0), Fmax - F)            (Fmax <= 0: unbounded)
 *   through   p_t = I_t - c_t
 *   evacuate  mode 0 (capacity):  e_t = min(F, max(Qe - p_t, 0))       release up to the target Qe
 *             mode 1 (linear):    e_t = (p_t < Qc) ? F / (Te + 1) : 0  implicit linear drain, gated off in floods
 *             mode 2 (min):       e_t = min(F / (Te + 1), max(Qe - p_t, 0))
 *   F += c_t - e_t
 * Rule-curve flux r_t (optional): the L4 zero-mean harmonic flux of rulecurve_offline/rc.py.
 * Withdrawal (optional): W_t = min(W*_t, max(S + p_t + e_t - r_t, 0)), taken from what the dam holds today,
 *   leaves the system (reported in Wout).
 * Engine form: c_t, e_t, W_t are computed from step-t state (F_t, S_t = T Q_t, the dam's step-t inflow) and
 *   subtracted from / added to the dam row's q' before the solve; F is updated after the solve.
 * Returns the SSE over mask (obs may be NULL). Q, Cout, Eout, Wout, Fout, fl may be NULL.
 */
double law(const double *I, const double *r, const double *Wstar, int n, double T,
           int mode, double Qc, double phi, double Fmax, double Qe, double Te, int floor_on,
           const double *obs, const unsigned char *mask,
           double *Q, double *Cout, double *Eout, double *Wout, double *Fout, unsigned char *fl)
{
    double F = 0.0, S = T * I[0], sse = 0.0;
    for (int t = 0; t < n; t++) {
        double c = 0.0, e = 0.0, p = I[t];
        if (mode >= 0) {
            double ex = I[t] - Qc;
            c = ex > 0.0 ? phi * ex : 0.0;
            if (Fmax > 0.0 && c > Fmax - F) c = Fmax - F > 0.0 ? Fmax - F : 0.0;
            p = I[t] - c;
            if (mode == 0) {
                double h = Qe - p;
                e = h > 0.0 ? (h < F ? h : F) : 0.0;
            } else if (mode == 1) {
                e = (p < Qc) ? F / (Te + 1.0) : 0.0;
            } else {
                double h = Qe - p, lin = F / (Te + 1.0);
                e = h > 0.0 ? (h < lin ? h : lin) : 0.0;
            }
            F += c - e;
        }
        double x = p + e - (r ? r[t] : 0.0);
        double w = 0.0;
        if (Wstar) {
            double av = S + x;
            w = Wstar[t] < av ? Wstar[t] : av;
            if (w < 0.0) w = 0.0;
            x -= w;
        }
        double avail = S + x;
        double q = avail / (T + 1.0);
        unsigned char f = 0;
        if (floor_on && q < 0.0) { q = 0.0; f = 1; }
        S = avail - q;
        if (Q) Q[t] = q;
        if (Cout) Cout[t] = c;
        if (Eout) Eout[t] = e;
        if (Wout) Wout[t] = w;
        if (Fout) Fout[t] = F;
        if (fl) fl[t] = f;
        if (mask && mask[t]) { double d = q - obs[t]; sse += d * d; }
    }
    return sse;
}
