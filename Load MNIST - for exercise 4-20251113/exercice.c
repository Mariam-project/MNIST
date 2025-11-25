#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "load_mnist.h"

#define INPUT_SIZE  (28*28) //784
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 10

//Fonction qui choisit une fonction aléatoire 
void motif_aleatoire() {
    image img; // Initialisation d'une struct image
    int index = rand() % 59000; // de 0 à 59000
    read_training_image(index, &img);

    printf("Motif tiré : %d\n", index);
    affiche_img(&img);
}

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
        X[j] = img->imgbuf[j] / 255.0f;
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






//---------------------------------- MAIN -------------------------------------
int main() {

    open_training_files();

    image img;
    int Yd[10];      // vecteur one-hot
    float X[784];    // rétine normalisée

    // 1. Charger un motif au hasard + créer Yd
    get_random_training_image(&img, Yd);

    // 2. Normaliser l'image
    propager_sur_retine(&img, X);
    
    // Déclaration des poids et des couches
    float W1[HIDDEN_SIZE][INPUT_SIZE];
    float W2[OUTPUT_SIZE][HIDDEN_SIZE];
    float Xh[HIDDEN_SIZE];   // couche cachée
    float Xi[OUTPUT_SIZE];   // couche de sortie

    initialiser_poids(W1, W2);

    // 3. Calcul couche cachée
    calcul_couche_cachee(X, W1, Xh);

    // 4. Calcul couche sortie
    calcul_couche_sortie(Xh, W2, Xi);

    // Affichage des sorties
    printf("Sortie réseau :\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        printf("Neurone %d : %f\n", i, Xi[i]);


    // Vérification
    affiche_img(&img);

    close_training_files();
    return 0;
}

