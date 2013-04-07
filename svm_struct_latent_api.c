/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct                     */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 17.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include "svm_struct_latent_api_types.h"
#include <limits.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

#define MAX_INPUT_LINE_LENGTH 10000
#define EQUALITY_EPSILON 1e-6

void die(const char *message)
{
  if(errno) {
      perror(message); 
  } else {
      printf("ERROR: %s\n", message);
  }
  exit(1);
}

int img_score_comp(const void *a, const void *b) {
    IMG_SCORE *aa = (IMG_SCORE*)a;
    IMG_SCORE *bb = (IMG_SCORE*)b;

    if(aa->img_score > bb->img_score)
        return -1;
    if(bb->img_score > aa->img_score)
        return 1;
    return (aa->img_idx > bb->img_idx) ? 1 : -1;   // Cannot compare equal.
}

SVECTOR** readFeatures(char *feature_file, int n_fvecs) {
    WORD *words = NULL;
    FILE *fp = fopen(feature_file, "r");
    
    int fvec_length = 0;
    char line[1000000];
    size_t ln;
    char *pair, *single, *brkt, *brkb;

    SVECTOR **fvecs = (SVECTOR **)malloc(n_fvecs*sizeof(SVECTOR *));
    if(!fvecs) die("Memory Error.");

    int i = 0;
    int j = 0;

    while( fgets(line,1000000,fp) ) {
        ln = strlen(line) - 1;
        if (line[ln] == '\n')
            line[ln] = '\0';
        
        fvec_length = 0;
        for(pair = strtok_r(line, " ", &brkt); pair; pair = strtok_r(NULL, " ", &brkt)){
            fvec_length++;
            words = (WORD *) realloc(words, fvec_length*sizeof(WORD));
            if(!words) die("Memory error.");
            j = 0;
            for (single = strtok_r(pair, ":", &brkb); single; single = strtok_r(NULL, ":", &brkb)){
                if(j == 0){
                    words[fvec_length-1].wnum = atoi(single);
                }
                else{
                    words[fvec_length-1].weight = atof(single); 
                }
                j++;
            }
        }   
        fvec_length++; 
        words = (WORD *) realloc(words, fvec_length*sizeof(WORD));
        if(!words) die("Memory error.");
        words[fvec_length-1].wnum = 0;
        words[fvec_length-1].weight = 0.0;

        fvecs[i] = create_svector(words,"",1);
        free(words);
        words = NULL;
        i++;
   }
   return fvecs;
}

SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) {
    SAMPLE sample;

    int i , j; 
    
    // open the file containing candidate bounding box dimensions/labels/featurePath and image label
    FILE *fp = fopen(file, "r");
    if(fp==NULL){
        printf("Error: Cannot open input file %s\n",file);
        exit(1);      
    }

    // We treat all images as one example, since there is only one correct structural output Y* for all images combined
    sample.n = 1;  
    sample.examples = (EXAMPLE *) malloc(sample.n*sizeof(EXAMPLE));
    if(!sample.examples) die("Memory error.");
    sample.examples[0].n_pos = 0;
    sample.examples[0].n_neg = 0;
    
    fscanf(fp,"%ld", &sample.examples[0].n_imgs);
    
    /* Initialise pattern */
    sample.examples[0].x.example_cost = 1;
    sample.examples[0].x.x_is = (SUB_PATTERN *) malloc(sample.examples[0].n_imgs*sizeof(SUB_PATTERN));
    if(!sample.examples[0].x.x_is) die("Memory error.");
    sample.examples[0].y.labels = (int *) malloc(sample.examples[0].n_imgs*sizeof(int));
    if(!sample.examples[0].y.labels) die("Memory error.");
    
    for(i = 0; i < sample.examples[0].n_imgs; i++){
        fscanf(fp, "%s", sample.examples[0].x.x_is[i].file_name);
        fscanf(fp,"%d",&sample.examples[0].x.x_is[i].label);
        fscanf(fp, "%d", &sample.examples[0].x.x_is[i].n_candidates);
        
        sample.examples[0].y.labels[i] = sample.examples[0].x.x_is[i].label;
        // Image label can be 0(negative image) or 1(positive image)
        if(sample.examples[0].x.x_is[i].label == 0) {
            sample.examples[0].n_neg++;
        } else { 
            sample.examples[0].n_pos++;
            sample.examples[0].x.x_is[i].areaRatios = (int *) malloc (sample.examples[0].x.x_is[i].n_candidates*sizeof(int));
            if(!sample.examples[0].x.x_is[i].areaRatios) die("Memory error.");
            for(j = 0; j < sample.examples[0].x.x_is[i].n_candidates; j++){
                fscanf(fp, "%d", &sample.examples[0].x.x_is[i].areaRatios[j]);
            }    
        }

        sample.examples[0].x.x_is[i].phis = (SVECTOR **) malloc(sample.examples[0].x.x_is[i].n_candidates*sizeof(SVECTOR *));
        if(!sample.examples[0].x.x_is[i].phis) die("Memory error.");  
    }
     
    sample.examples[0].x.n_pos = sample.examples[0].n_pos;
    sample.examples[0].x.n_neg = sample.examples[0].n_neg;
    sample.examples[0].y.n_pos = sample.examples[0].n_pos;
    sample.examples[0].y.n_neg = sample.examples[0].n_neg;
    
    /* Intialise label*/
    sample.examples[0].y.ranking = (int *) calloc((sample.examples[0].n_pos+sample.examples[0].n_neg), sizeof(int));
    for(i = 0; i < (sample.examples[0].n_pos+sample.examples[0].n_neg); i++){
        for(j = i+1; j < (sample.examples[0].n_pos+sample.examples[0].n_neg); j++){
            if(sample.examples[0].x.x_is[i].label == 1){
                if(sample.examples[0].x.x_is[j].label == 0){
                    sample.examples[0].y.ranking[i]++;
                    sample.examples[0].y.ranking[j]--;
                }              
            }
            else{
                if(sample.examples[0].x.x_is[j].label == 1){
                    sample.examples[0].y.ranking[i]--;
                    sample.examples[0].y.ranking[j]++;
                }
            }
        }
    }

    return sample;
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the diminension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

	sm->n = sample.n;
	sm->sizePsi = sparm->feature_size;

}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/
    sample->examples[0].h.h_is = (int *) malloc((sample->examples[0].n_pos+sample->examples[0].n_neg)*sizeof(int));
    sample->examples[0].h.phi_h_is = (SVECTOR **) malloc((sample->examples[0].n_pos+sample->examples[0].n_neg)*sizeof(SVECTOR *));

    long i;
    int positive_candidate;
    SVECTOR **fvecs = NULL;

    int j;

    srand(sparm->rng_seed);
    for(i=0; i < (sample->examples[0].n_pos+sample->examples[0].n_neg); i++){
        sample->examples[0].h.phi_h_is[i] = NULL;
        if(sample->examples[0].x.x_is[i].label == 1){
            positive_candidate = (int) (((float)sample->examples[0].x.x_is[i].n_candidates)*((float)rand())/(RAND_MAX+1.0)); 
            sample->examples[0].h.h_is[i] = positive_candidate; 
            
            fvecs = readFeatures(sample->examples[0].x.x_is[i].file_name, sample->examples[0].x.x_is[i].n_candidates);
            sample->examples[0].h.phi_h_is[i] = copy_svector(fvecs[sample->examples[0].h.h_is[i]]);
            for (j = 0; j < sample->examples[0].x.x_is[i].n_candidates; j++){
                free_svector(fvecs[j]);
            }
            free(fvecs);
        }
        else{
            sample->examples[0].h.h_is[i] = -1; // There is no latent variable for negative samples.
        }                       
    }
	
}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/
    SVECTOR *fvec=NULL;
    
    // your code here 
  
    long i;
    long j;
    
    double Y_ij;
    double norm_factor;
    int y_i_count;
    
    SVECTOR *temp1=NULL;
    SVECTOR *temp2=NULL;
    SVECTOR *temp3=NULL;
    SVECTOR *temp4=NULL;
    SVECTOR *temp5=NULL;
    
    WORD *words = (WORD *) malloc(sizeof(WORD));
	words[0].wnum = 0;
	words[0].weight = 0.0;
	fvec = create_svector(words,"",1);
	free(words);
	
	int *coeff = (int*) calloc((x.n_pos + x.n_neg), sizeof(int));

    for(i = 0; i < (x.n_pos + x.n_neg); i++){
        if(x.x_is[i].label == 1){  
            y_i_count = 0; 
            
            for(j = 0; j < (x.n_pos + x.n_neg); j++){
                if(x.x_is[j].label == 0){
                    if(y.ranking[i] > y.ranking[j]){
                        Y_ij = 1;
                    }
                    else{
                        Y_ij = -1;
                    }
                    if(Y_ij == 1){
                        coeff[j]--;              
                        y_i_count++;
                    }
                    else{
                        coeff[j]++;                      
                        y_i_count--;
                    }                                                      
                }
            }
            temp2 = smult_s(h.phi_h_is[i], y_i_count);
            temp3 = add_ss(fvec, temp2);
            free_svector(temp2);
            free_svector(fvec);
            fvec = temp3;
        }        
    }
    
    for(j = 0; j < (x.n_pos + x.n_neg); j++){
        if(x.x_is[j].label == 0){
            temp1 = smult_s(h.phi_h_is[j], coeff[j]);
            temp5 = add_ss(fvec, temp1);
            free_svector(temp1);
            free_svector(fvec);
            fvec = temp5; 
        }
    }

    norm_factor = 1/(double)(x.n_pos*x.n_neg);
    temp4 = smult_s(fvec, norm_factor);
    free(fvec);
    fvec = temp4;
     
    return(fvec);
}

double *classify_struct_example(PATTERN x, STRUCTMODEL *sm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
    long i;
    double *scores = (double *) malloc((x.n_pos+x.n_neg)*sizeof(double));
    
    for(i = 0; i < (x.n_neg+x.n_pos); i++){
        scores[i] = sprod_ns(sm->w, x.x_is[i].phis[0]); // There isn't any latent var for test images, so we use 0 index        
    }

	return scores;
}

void encodeRanking(PATTERN x, LABEL *ybar, IMG_SCORE *positiveImgScores, IMG_SCORE *negativeImgScores, int *imgIndexMap, int *optimumLocNegImg){
    int i, j, i_prime, j_prime, oj_prime, oi_prime;
    
    ybar->ranking = (int *) calloc((x.n_pos+x.n_neg), sizeof(int));
    if(!ybar->ranking) die("Memory error");
    
    for(i = 0; i < (x.n_pos+x.n_neg); i++){
        for(j = i+1; j < (x.n_pos+x.n_neg); j++){        
            if(i == j){
                //ybar->rank_matrix[i][j] = 0;
            }
            else if(x.x_is[i].label == x.x_is[j].label){                
                if(x.x_is[i].label == 1){                                 
                    if(positiveImgScores[imgIndexMap[i]].img_score > positiveImgScores[imgIndexMap[j]].img_score){
                        //ybar->rank_matrix[i][j] = 1;
                        ybar->ranking[i]++;
                        ybar->ranking[j]--;
                    }
                    else if(positiveImgScores[imgIndexMap[j]].img_score > positiveImgScores[imgIndexMap[i]].img_score){
                        //ybar->rank_matrix[i][j] = -1;
                        ybar->ranking[i]--;
                        ybar->ranking[j]++;
                    }
                    else{
                        if(i < j){
                            //ybar->rank_matrix[i][j] = 1;
                            ybar->ranking[i]++;
                            ybar->ranking[j]--;
                        }
                        else{
                            //ybar->rank_matrix[i][j] = -1;
                            ybar->ranking[i]--;
                            ybar->ranking[j]++;
                        }
                    }
                }
                else{
                    if(negativeImgScores[imgIndexMap[i]].img_score > negativeImgScores[imgIndexMap[j]].img_score){
                        //ybar->rank_matrix[i][j] = 1;
                        ybar->ranking[i]++;
                        ybar->ranking[j]--;
                    }
                    else if(negativeImgScores[imgIndexMap[j]].img_score > negativeImgScores[imgIndexMap[i]].img_score){
                        //ybar->rank_matrix[i][j] = -1;
                        ybar->ranking[i]--;
                        ybar->ranking[j]++;
                    }
                    else{
                        if(i < j){
                            //ybar->rank_matrix[i][j] = 1;
                            ybar->ranking[i]++;
                            ybar->ranking[j]--;
                        }
                        else{
                            //ybar->rank_matrix[i][j] = -1;
                            ybar->ranking[i]--;
                            ybar->ranking[j]++;
                        }
                    }
                }        
            }
            else if((x.x_is[i].label == 1) && (x.x_is[j].label == 0)){
                i_prime = imgIndexMap[i]+1;
                j_prime = imgIndexMap[j]+1;
                oj_prime = optimumLocNegImg[j_prime-1];
                              
                if((oj_prime - i_prime - 0.5) > 0){
                    //ybar->rank_matrix[i][j] = 1;
                    ybar->ranking[i]++;
                    ybar->ranking[j]--;
                }
                else{
                    //ybar->rank_matrix[i][j] = -1;
                    ybar->ranking[i]--;
                    ybar->ranking[j]++;
                }
            }
            else if((x.x_is[i].label == 0) && (x.x_is[j].label == 1)){
                i_prime = imgIndexMap[i]+1;
                j_prime = imgIndexMap[j]+1;
                oi_prime = optimumLocNegImg[i_prime-1];
                
                if((j_prime - oi_prime + 0.5) > 0){
                    //ybar->rank_matrix[i][j] = 1;
                    ybar->ranking[i]++;
                    ybar->ranking[j]--;
                }
                else{
                    //ybar->rank_matrix[i][j] = -1;
                    ybar->ranking[i]--;
                    ybar->ranking[j]++;
                }
            }                    
        }        
    }    
}

void findOptimumNegLocations(PATTERN x, LABEL *ybar, IMG_SCORE *positiveImgScores, IMG_SCORE *negativeImgScores, int *imgIndexMap){
    int j, k;
    int *optimumLocNegImg = malloc(x.n_neg*sizeof(*optimumLocNegImg));
    if(!optimumLocNegImg) die("Memory error");

    double maxValue = 0;
    double currentValue = 0;
    int maxIndex = -1;
    // for every jth negative image
    for(j = 1; j <= x.n_neg; j++){
        maxValue = 0;
        maxIndex = x.n_pos+1;
        // k is what we are maximising over. There would be one k_max for each negative image j
        currentValue = 0;
        for(k = x.n_pos; k >= 1; k--){
                currentValue += (1/(double)x.n_pos)*((j/(double)(j+k))-((j-1)/(double)(j+k-1))) - (2/(double)(x.n_pos*x.n_neg))*(positiveImgScores[k-1].img_score - negativeImgScores[j-1].img_score);
            if(currentValue > maxValue){
                maxValue = currentValue;
                maxIndex = k;
            }
        }
        optimumLocNegImg[j-1] = maxIndex;
    }
    encodeRanking(x, ybar, positiveImgScores, negativeImgScores, imgIndexMap, optimumLocNegImg);
    free(optimumLocNegImg);
}

void find_most_violated_constraint_marginrescaling(PATTERN *x, LABEL y, LATENT_VAR *h, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
    int i, j;    
    
    double maxScore = -DBL_MAX;
    double score;

    SVECTOR **fvecs = NULL;
    
    for(i = 0; i < (x->n_pos+x->n_neg); i++){
        maxScore = -DBL_MAX;
        if(x->x_is[i].label == 0){
            if (h->phi_h_is[i]){
                free_svector(h->phi_h_is[i]);
            }
            fvecs = readFeatures(x->x_is[i].file_name, x->x_is[i].n_candidates);
            for(j = 0; j < x->x_is[i].n_candidates; j++){
                score = sprod_ns(sm->w, fvecs[j]);      
                if(score > maxScore){
                    maxScore = score;
                    h->h_is[i] = j;
                }   
            }
            h->phi_h_is[i] = copy_svector(fvecs[h->h_is[i]]);
            for(j =0; j < x->x_is[i].n_candidates; j++){
                free_svector(fvecs[j]);
            }
            free(fvecs);
        }
    }

    hbar->h_is = malloc((x->n_pos+x->n_neg)*sizeof(int));
    if(!hbar->h_is) die("Memory error");
    hbar->phi_h_is = (SVECTOR **) malloc((x->n_pos+x->n_neg)*sizeof(SVECTOR *));

    for(i = 0; i < (x->n_pos+x->n_neg); i++){
        hbar->h_is[i] = h->h_is[i];
        hbar->phi_h_is[i] = copy_svector(h->phi_h_is[i]);
    }
        
    IMG_SCORE *positiveImgScores = malloc(x->n_pos*sizeof(IMG_SCORE));
    if(!positiveImgScores) die("Memory error");
    IMG_SCORE *negativeImgScores = malloc(x->n_neg*sizeof(IMG_SCORE));    
    if(!negativeImgScores) die("Memory error");
    int *imgIndexMap = malloc((x->n_pos+x->n_neg)*sizeof(*imgIndexMap));
    if(!imgIndexMap) die("Memory error");
    int negativeId = 0;
    int positiveId = 0;    
    // find scores of all positive images
    for(i = 0; i < (x->n_pos + x->n_neg); i++){
        score = sprod_ns(sm->w, h->phi_h_is[i]);
        if(x->x_is[i].label == 0){
            negativeImgScores[negativeId].img_idx = i;
            negativeImgScores[negativeId].img_score = score;
            negativeId++;
        }
        else{
            positiveImgScores[positiveId].img_idx = i;
            positiveImgScores[positiveId].img_score = score;
            positiveId++;
        }
    }
    
    // sort positiveImgScores and negativeImgScores in descending order of score
    qsort(negativeImgScores, x->n_neg, sizeof(IMG_SCORE), img_score_comp);
    qsort(positiveImgScores, x->n_pos, sizeof(IMG_SCORE), img_score_comp);
    
    negativeId = 0;
    positiveId = 0; 
    for(i = 0; i < (x->n_pos+x->n_neg); i++){
        if(x->x_is[i].label == 1){
            imgIndexMap[positiveImgScores[positiveId].img_idx] = positiveId;
            positiveId++;
        }
        else{
            imgIndexMap[negativeImgScores[negativeId].img_idx] = negativeId;
            negativeId++;
        }
    }    
    
    findOptimumNegLocations(*x, ybar, positiveImgScores, negativeImgScores, imgIndexMap);
    ybar->n_pos = x->n_pos;
    ybar->n_neg = x->n_neg;
    ybar->labels = malloc((x->n_pos+x->n_neg)*sizeof(int));
    
    free(positiveImgScores);
    free(negativeImgScores);
    free(imgIndexMap);
}

void infer_latent_variables(PATTERN x, LABEL y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  //LATENT_VAR h;
  
      /* your code here */
    long i;
    int j;
    
   
    //h->h_is = (int *) malloc((x.n_pos+x.n_neg)*sizeof(int));
    double maxScore = -DBL_MAX;
    double curr_score;
    
    SVECTOR **fvecs = NULL;

    for(i = 0; i < (x.n_pos+x.n_neg); i++){
        maxScore = -DBL_MAX;
        if(x.x_is[i].label == 1){
            free_svector(h->phi_h_is[i]);
            fvecs = readFeatures(x.x_is[i].file_name, x.x_is[i].n_candidates);
            for(j = 0; j < x.x_is[i].n_candidates; j++){
                //if(s.x_is[i].isConsider){
                curr_score = sprod_ns(sm->w, fvecs[j]);      
                if(curr_score > maxScore){
                    maxScore = curr_score;
                    h->h_is[i] = j;
                }   
                //}                
            }
            h->phi_h_is[i] = copy_svector(fvecs[h->h_is[i]]);
            for(j =0; j < x.x_is[i].n_candidates; j++){
                free_svector(fvecs[j]);
            }
            free(fvecs);
        }
    }

    //return(h); 

}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
/*
  Computes the loss of prediction (ybar,hbar) against the
  correct label y. 
*/ 
	double l;
	
	long i;
    long j;

    int *ranking = malloc((y.n_pos+y.n_neg)*sizeof(int)); // stores rank of all images
    int *sortedImages = malloc((y.n_pos+y.n_neg)*sizeof(int)); // stores list of images sorted by rank. Higher rank to lower rank 
    
    /* convert rank matrix to rank list*/
    for(i = 0; i < (y.n_pos+y.n_neg); i++){
        ranking[i] = 1; // start with lowest rank for each sample i.e 1 
        for(j = 0; j < (y.n_pos+y.n_neg); j++){
            if(ybar.ranking[i] > ybar.ranking[j]){
                ranking[i] = ranking[i] + 1;
            } 
        }
        sortedImages[(y.n_pos+y.n_neg)-ranking[i]] = i;
    }  
    
    int posCount = 0;
    int totalCount = 0;
    double precisionAti = 0;
    int label;
    for(i = 0; i < (y.n_pos+y.n_neg); i++){
        label = y.labels[sortedImages[i]];
        if(label == 1){
            posCount++;
            totalCount++;
        }
        else{
            totalCount++;
        }
        if(label == 1){
            precisionAti = precisionAti + (double)posCount/(double)totalCount;
        }
    }
    precisionAti = precisionAti/(double)posCount;
    
    l = 1 - precisionAti;
    
    free(ranking);
    free(sortedImages);

	return(l);

}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Writes the learned weight vector sm->w to file after training. 
*/
  FILE *modelfl;
  int i;
  
  modelfl = fopen(file,"w");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for output!", file);
		exit(1);
  }
  
  for (i=1;i<sm->sizePsi+1;i++) {
    fprintf(modelfl, "%d:%.16g\n", i, sm->w[i]);
  }
  fclose(modelfl);
 
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Reads in the learned model parameters from file into STRUCTMODEL sm.
  The input file format has to agree with the format in write_struct_model().
*/
  STRUCTMODEL sm;

  FILE *modelfl;
  int sizePsi,i, fnum;
  double fweight;
  
  modelfl = fopen(file,"r");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for input!", file);
	exit(1);
  }

	sizePsi = 1;
	sm.w = (double*)malloc((sizePsi+1)*sizeof(double));
	for (i=0;i<sizePsi+1;i++) {
		sm.w[i] = 0.0;
	}
	while (!feof(modelfl)) {
		fscanf(modelfl, "%d:%lf", &fnum, &fweight);
		if(fnum > sizePsi) {
			sizePsi = fnum;
			sm.w = (double *)realloc(sm.w,(sizePsi+1)*sizeof(double));
		}
		sm.w[fnum] = fweight;
	}

	fclose(modelfl);

	sm.sizePsi = sizePsi;

  return(sm);

}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/

  free(sm.w);

}

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/

     int i;

    for(i = 0; i < (x.n_pos+x.n_neg); i++){
        free(x.x_is[i].areaRatios);
        free(x.x_is[i].phis);
    }  
    free(x.x_is);

}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/
    /* your code here */
    
    /*for(i = 0; i < (y.n_pos + y.n_neg); i++){
        free(y.rank_matrix[i]);
    } */  
    //free(y.rank_matrix);
    free(y.ranking);
    free(y.labels);

} 

void free_latent_var(LATENT_VAR h, PATTERN x) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/
    int i;
    free(h.h_is);
    for(i = 0; i < (x.n_pos+x.n_neg); i++){
        free_svector(h.phi_h_is[i]);
    }
    free(h.phi_h_is);
}

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
  int i;
  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h, s.examples[i].x);
  }
  free(s.examples);

}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;
  
  /* set default */
  sparm->feature_size = 90112;
  sparm->rng_seed = 0;
  sparm->learning_type = 0; // default learning type set to 0, corresponding to unpooled negatives
  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
      case 'f': i++; sparm->feature_size = atoi(sparm->custom_argv[i]); break;
      case 'r': i++; sparm->rng_seed = atoi(sparm->custom_argv[i]); break;
      case 't': i++; sparm->learning_type = atoi(sparm->custom_argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }

}

void copy_label(LABEL l1, LABEL *l2)
{
}

void copy_latent_var(LATENT_VAR lv1, LATENT_VAR *lv2)
{
}

void print_latent_var(LATENT_VAR h, FILE *flatent)
{
}

void print_label(LABEL l, FILE	*flabel)
{
}
