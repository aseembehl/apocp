/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api_types.h                                      */
/*                                                                      */
/*   API type definitions for Latent SVM^struct                         */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 30.Sep.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

# include "svm_light/svm_common.h"

typedef struct imgScore{
    int img_idx;
    double img_score;
    //int img_h;
} IMG_SCORE;


typedef struct sub_pattern {
  /*
    Type definition for input pattern x
  */
  char file_name[1000];
  int *areaRatios;
  SVECTOR **phis;
  int n_candidates;
  int label;
    
} SUB_PATTERN;

typedef struct pattern {
  /*
    Type definition for input pattern x
  */
    double example_cost; /* cost of individual example */

    SUB_PATTERN *x_is;
    long n_pos;
    long n_neg;
    long n_neg_boxes;
    long n_unsup_neg;
} PATTERN;

typedef struct label {
  /*
    Type definition for output label y
  */
  //int **rank_matrix;
  int *ranking;
  int *labels;
  
  long n_pos;
  long n_neg;
} LABEL;

typedef struct _sortStruct {
    double val;
    int index;
}  sortStruct;

typedef struct latent_var {
  /*
    Type definition for latent variable h
  */
  int *h_is;
  SVECTOR **phi_h_is;
} LATENT_VAR;

typedef struct example {
  PATTERN x;
  LABEL y;
  LATENT_VAR h;
  
  long n_imgs; 
  long n_pos;
  long n_neg;

} EXAMPLE;

typedef struct sample {
  int n;
  EXAMPLE *examples;

} SAMPLE;


typedef struct structmodel {
  double *w;          /* pointer to the learned weights */
  MODEL  *svm_model;  /* the learned SVM model */
  long   sizePsi;     /* maximum number of weights in w */
  /* other information that is needed for the stuctural model can be
     added here, e.g. the grammar rules for NLP parsing */
  long n;             /* number of examples */
} STRUCTMODEL;


typedef struct struct_learn_parm {
  double epsilon;              /* precision for which to solve
				  quadratic program */
  long newconstretrain;        /* number of new constraints to
				  accumulate before recomputing the QP
				  solution */
  double C;                    /* trade-off between margin and loss */
  char   custom_argv[20][1000]; /* string set with the -u command line option */
  int    custom_argc;          /* number of -u command line options */
  int    slack_norm;           /* norm to use in objective function
                                  for slack variables; 1 -> L1-norm, 
				  2 -> L2-norm */
  int    loss_type;            /* selected loss function from -r
				  command line option. Select between
				  slack rescaling (1) and margin
				  rescaling (2) */
  int    loss_function;        /* select between different loss
				  functions via -l command line
				  option */
  /* add your own variables */
  long feature_size;
  int rng_seed;
  int isInitByBinSVM;
  int initIter;

  int learning_type;
  int min_area_ratios[6];
  
} STRUCT_LEARN_PARM;

