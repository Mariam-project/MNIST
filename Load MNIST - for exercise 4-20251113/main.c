#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "load_mnist.h"



char gl[10] = {' ','.',':','-','=','+','*','#','%','@'};
char bigl[96] = 
"@MBHENR#KWXDFPQASUZbdehx*8Gm&04LOVYkpq5Tagns69owz$CIu23Jcfry%1v7l+it[]{}?j|()=~!-/<>^^^_';,:`. ";




void print_image(image* img){
int i;
char grey_level;

/* print the image "pixels" in grey_level using ascii*/
for (i=0;i<= __SIZE_IMAGE* __SIZE_IMAGE;i++){
        /* grey scale ascii art*/
	
	grey_level=(img->imgbuf[i])/2.7;
	printf("%c ",bigl[grey_level]);

	if (i%__SIZE_IMAGE==0) printf("\n");
	}

/* print the label */

printf("label : %d \n",img->label);

}




int main() {
srand(time(0));
image ret; 
open_training_files();

read_training_image(rand()%60000,&ret);
print_image(&ret);

close_training_files();
return 0;
}
