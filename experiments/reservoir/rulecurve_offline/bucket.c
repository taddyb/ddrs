/* Daily linear bucket, the harness recursion (implicit Euler on storage, same-day inflow):
 *   avail = S + x_t ; q_t = avail / (T_t + 1) ; [floor: q_t = max(q_t, 0)] ; S = avail - q_t
 * x_t is the effective inflow I_t - r_t (r = rule-curve flux). With the floor on, a floored day releases
 * nothing and the whole (possibly negative) avail stays in storage, so mass is conserved in the bucket state.
 * Initial storage S = T_0 * x_0 (the harness's S = T * I[0]).
 * Returns the SSE over days t < n with mask[t] != 0 (mask may be NULL); Q and fl may be NULL. */
double bucket(const double *x, const double *T, int n, int floor_on, const double *obs,
              const unsigned char *mask, double *Q, unsigned char *fl)
{
    double S = T[0] * x[0];
    double sse = 0.0;
    for (int t = 0; t < n; t++) {
        double avail = S + x[t];
        double q = avail / (T[t] + 1.0);
        unsigned char f = 0;
        if (floor_on && q < 0.0) { q = 0.0; f = 1; }
        /* floor_on == 1: the floored deficit stays in storage (S = avail - q, may go negative).
         * floor_on == 2: engine-like clamp, the closed form S = T q is kept (S = 0 on a floored day), so the
         *                deficit is forgotten and mass is created (what a post-solve clamp_min on Q does). */
        S = (floor_on == 2 && f) ? 0.0 : avail - q;
        if (Q) Q[t] = q;
        if (fl) fl[t] = f;
        if (mask && mask[t]) { double e = q - obs[t]; sse += e * e; }
    }
    return sse;
}
