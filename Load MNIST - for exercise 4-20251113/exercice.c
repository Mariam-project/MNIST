#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "load_mnist.h"

#define INPUT_SIZE  (28*28) //784
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 10

// //Fonction qui choisit un motif aléatoire 
// void motif_aleatoire() {
//     image img; // Initialisation d'une struct image
//     int index = rand() % 59000; // de 0 à 59000
//     read_training_image(index, &img); // fonction existe déjà

//     printf("Motif tiré : %d\n", index);
//     affiche_img(&img);
// }

//Fonction pour initialiser les poids 
void initialiser_poids(float W1[HIDDEN_SIZE][INPUT_SIZE],
                       float W2[OUTPUT_SIZE][HIDDEN_SIZE]) {
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            W1[i][j] = ((float)rand()/RAND_MAX - 0.5f) / 10.0f;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            W2[i][j] = ((float)rand()/RAND_MAX - 0.5f) / 10.0f;
}

// --- Fonction : tirer un motif au hasard et le charger ------------------
void get_random_training_image(image *img, int Yd[10]) {
    // choisir un index entre 0 et 59999
    int index = rand() % 60000;

    // charger l'image correspondante
    read_training_image(index, img);

    // initialiser Yd à 0
    for (int i = 0; i < 10; i++)
        Yd[i] = 0;

    // définir la classe correcte 
    Yd[img->label] = 1;
}

// --- Fonction : normaliser les pixels dans X[784] -----------------------
void propager_sur_retine(const image *img, float X[784]) {
    for (int j = 0; j < 784; j++) {
        X[j] = img->imgbuf[j] / 255.0f; //imgbuf c'est dans la structure 
    }
}

// Fonction d'activation : f(x) = 1 / (1 + e^{-x})
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Calcul de la sortie des neurones de la couche cachée
void calcul_couche_cachee(float X[INPUT_SIZE],
                          float W1[HIDDEN_SIZE][INPUT_SIZE],
                          float Xh[HIDDEN_SIZE])
{
    for (int h = 0; h < HIDDEN_SIZE; h++) {

        float somme = 0.0f;  // poth

        // somme pondérée : Σ_j W1[h][j] * X[j]
        for (int j = 0; j < INPUT_SIZE; j++)
            somme += W1[h][j] * X[j];

        Xh[h] = sigmoid(somme); // activation
    }
}

// Calcul de la couche de sortie
void calcul_couche_sortie(float Xh[HIDDEN_SIZE],
                          float W2[OUTPUT_SIZE][HIDDEN_SIZE],
                          float Xi[OUTPUT_SIZE])
{
    for (int i = 0; i < OUTPUT_SIZE; i++) {

        float somme = 0.0f;  // poti

        // somme pondérée : Σ_h W2[i][h] * Xh[h]
        for (int h = 0; h < HIDDEN_SIZE; h++)
            somme += W2[i][h] * Xh[h];

        Xi[i] = sigmoid(somme); // activation finale
    }
}

// ---------------------------------------------------------------------------
// 5. Calcul du delta de la couche de sortie
// delta_i = f'(poti) * (Yd_i - Xi)
// avec f'(x) = f(x)*(1 - f(x)) car f = sigmoïde
// ---------------------------------------------------------------------------
void calcul_delta_sortie(float X_out[OUTPUT_SIZE],  // sorties Xi après sigmoïde
                         int Yd[OUTPUT_SIZE],       // vecteur one-hot
                         float delta_out[OUTPUT_SIZE])
{
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float Xi = X_out[i];              // sortie du neurone i
        float fprime = Xi * (1 - Xi);     // f'(poti)
        delta_out[i] = fprime * (Yd[i] - Xi);
    }
}


// ---------------------------------------------------------------------------
// 6. Calcul du delta de la couche cachée
// delta_h = f'(poth) * Σ_i (delta_i * W2[i][h])
// où f'(x) = Xh * (1 - Xh) car Xh = f(poth)
// ---------------------------------------------------------------------------
void calcul_delta_cachee(float X_hidden[HIDDEN_SIZE],     // sorties Xh
                         float delta_out[OUTPUT_SIZE],     // deltas couche sortie
                         float W2[OUTPUT_SIZE][HIDDEN_SIZE], // poids sortie
                         float delta_hidden[HIDDEN_SIZE])
{
    for (int h = 0; h < HIDDEN_SIZE; h++) {

        // 1) calcul du terme Σ_i (delta_i * W2[i][h])
        float somme = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            somme += delta_out[i] * W2[i][h];
        }

        // 2) f'(poth) = Xh * (1 - Xh)
        float Xh = X_hidden[h];
        float fprime = Xh * (1 - Xh);

        // 3) delta final
        delta_hidden[h] = fprime * somme;
    }
}

// ---------------------------------------------------------------------------
// Mise à jour des poids W1[h][j] (INPUT → HIDDEN)
// W1[h][j] += learning_rate * delta_hidden[h] * X[j]
// ---------------------------------------------------------------------------
void maj_poids_W1(float W1[HIDDEN_SIZE][INPUT_SIZE],
                  float delta_hidden[HIDDEN_SIZE],
                  float X[INPUT_SIZE],
                  float learning_rate)
{
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            W1[h][j] += learning_rate * delta_hidden[h] * X[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Mise à jour des poids W2[i][h] (HIDDEN → OUTPUT)
// W2[i][h] += learning_rate * delta_out[i] * X_hidden[h]
// ---------------------------------------------------------------------------
void maj_poids_W2(float W2[OUTPUT_SIZE][HIDDEN_SIZE],
                  float delta_out[OUTPUT_SIZE],
                  float X_hidden[HIDDEN_SIZE],
                  float learning_rate)
{
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            W2[i][h] += learning_rate * delta_out[i] * X_hidden[h];
        }
    }
}

// ---------------------------------------------------------------------------
// Calcul de l'erreur sur p images tirées au hasard
// Err = (1/p) * Σ_p Σ_i |delta_i|
// ---------------------------------------------------------------------------
float calcul_erreur(float W1[HIDDEN_SIZE][INPUT_SIZE],
                    float W2[OUTPUT_SIZE][HIDDEN_SIZE],
                    int p)
{
    float erreur_totale = 0.0f;

    for (int k = 0; k < p; k++) {

        image img;
        int Yd[10];
        float X[INPUT_SIZE];
        float Xh[HIDDEN_SIZE];
        float Xout[OUTPUT_SIZE];
        float deltaO[OUTPUT_SIZE];

        // 1) sélectionner une image
        get_random_training_image(&img, Yd);

        // 2) normaliser
        propager_sur_retine(&img, X);

        // 3) propagation avant
        calcul_couche_cachee(X, W1, Xh);
        calcul_couche_sortie(Xh, W2, Xout);

        // 4) calcul delta sortie
        calcul_delta_sortie(Xout, Yd, deltaO);

        // 5) ajouter |delta|
        for (int i = 0; i < OUTPUT_SIZE; i++)
            erreur_totale += fabs(deltaO[i]);
    }

    return erreur_totale / p;
}
// ------- TESTER LA PERFORMANCE DU RÉSEAU SUR LA BASE DE TEST -----
int prediction(float Xout[OUTPUT_SIZE]) {
    int maxi = 0;
    float maxv = Xout[0];

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (Xout[i] > maxv) {
            maxv = Xout[i];
            maxi = i;
        }
    }
    return maxi;
}

//--------------- fonction de test ----------

float tester_reseau(float W1[HIDDEN_SIZE][INPUT_SIZE],
                    float W2[OUTPUT_SIZE][HIDDEN_SIZE])
{
    open_test_files();

    int correct = 0;
    image img;
    float X[INPUT_SIZE];
    float Xh[HIDDEN_SIZE];
    float Xout[OUTPUT_SIZE];

    for (int k = 0; k < 10000; k++) {

        // 1. Lire l’image de test n°k
        read_test_image(k, &img);

        // 2. Normaliser
        propager_sur_retine(&img, X);

        // 3. Propagation avant
        calcul_couche_cachee(X, W1, Xh);
        calcul_couche_sortie(Xh, W2, Xout);

        // 4. Prédiction
        int pred = prediction(Xout);

        // 5. Comparaison
        if (pred == img.label)
            correct++;
    }

    close_test_files();

    return (correct / 10000.0f) * 100.0f; // pourcentage
}









//---------------------------------- MAIN -------------------------------------
int main() {

    open_training_files();

    // ---- Pour stocker les erreurs ----
    float erreurs[10000];
    int nb_points = 0;

    // ---- Lance Gnuplot ----
    FILE *gp = popen("gnuplot -persistent", "w");
    fprintf(gp, "set title 'Courbe d apprentissage'\n");
    fprintf(gp, "set xlabel 'Iteration (x1000)'\n");
    fprintf(gp, "set ylabel 'Erreur'\n");
    fprintf(gp, "set grid\n");
    fflush(gp);

    // Fichier pour stocker les erreurs
    FILE *ferr = fopen("erreur.dat", "w");


    // --- tableaux principaux ---
    float W1[HIDDEN_SIZE][INPUT_SIZE];
    float W2[OUTPUT_SIZE][HIDDEN_SIZE];

    float X[INPUT_SIZE];
    float Xh[HIDDEN_SIZE];
    float Xout[OUTPUT_SIZE];

    float deltaH[HIDDEN_SIZE];
    float deltaO[OUTPUT_SIZE];

    int Yd[10];
    image img;

    // ---- paramètres ----
    float learning_rate = 0.1;
    float seuil = 0.01;     // seuil d'erreur global
    int p = 100;            // nombre d'images pour évaluer l'erreur

    // 0) INITIALISATION
    initialiser_poids(W1, W2);

    // BOUCLE D'APPRENTISSAGE PRINCIPALE
    while (1) {

        // --- 1) Tirer une image ---
        get_random_training_image(&img, Yd);

        // --- 2) Rétine ---
        propager_sur_retine(&img, X);

        // --- 3) Propagation avant ---
        calcul_couche_cachee(X, W1, Xh);
        calcul_couche_sortie(Xh, W2, Xout);

        // --- 4) Delta sortie ---
        calcul_delta_sortie(Xout, Yd, deltaO);

        // --- 5) Delta cachée ---
        calcul_delta_cachee(Xh, deltaO, W2, deltaH);

        // --- 6-7) Mise à jour des poids ---
        maj_poids_W1(W1, deltaH, X, learning_rate);
        maj_poids_W2(W2, deltaO, Xh, learning_rate);

        // --- 8) Calculer erreur périodiquement ---
        static int compteur = 0;
        compteur++;
        if (compteur % 1000 == 0) {
            float Err = calcul_erreur(W1, W2, p);
            printf("Erreur actuelle = %.4f\n", Err);

            // Sauvegarde
            erreurs[nb_points] = Err;
            fprintf(ferr, "%d %.6f\n", nb_points, Err);
            fflush(ferr);
            nb_points++;

            // Mise à jour du plot
            fprintf(gp, "plot 'erreur.dat' with lines lw 2 title 'Erreur'\n");
            fflush(gp);

        if (Err < seuil)
                break;
}

    }

    printf("Apprentissage terminé !\n");
    // ---- Après la boucle d’apprentissage ----
    printf("Test du réseau sur la base de test...\n");

    float performance = tester_reseau(W1, W2);

    printf("Performance finale : %.2f %% de bonnes réponses\n", performance);
        
    fclose(ferr);
    pclose(gp);

    close_training_files();
    return 0;
}

