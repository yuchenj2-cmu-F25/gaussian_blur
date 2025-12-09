#ifndef CONFIG_H
#define CONFIG_H

// Allow command-line overrides via -DWIDTH=value
#ifndef WIDTH
#define WIDTH  512
#endif

#ifndef HEIGHT
#define HEIGHT 512
#endif

#ifndef RUNS
#define RUNS 100
#endif

#endif /* CONFIG_H */
