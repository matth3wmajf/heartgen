#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <alpha/feedforward.h>

int main(int argc, const char *argv[])
{
/* Set up the random number generator. */
	srand((unsigned int)time(NULL));

/*
 *	Define a variable for tracking the type of operation that we'll be
 *	performing.
 */
	enum
	{
		EXECUTION_TYPE_UNKNOWN,
		EXECUTION_TYPE_TRAIN,
		EXECUTION_TYPE_GENERATE
	} l_execution_type = EXECUTION_TYPE_UNKNOWN;

/* Pointers for pointing to the model and image file names. */
	char *pl_model_file_name = NULL;
	char *pl_image_file_name = NULL;

/* Parse the command-line argument(s). */
	for(uintmax_t l_i = 0; l_i < (uintmax_t)argc; l_i++)
	{
		if(strcmp("--train", argv[l_i]) == 0)
		{
/* ... */
			pl_model_file_name = (char *)argv[l_i + 1];
			pl_image_file_name = (char *)argv[l_i + 2];
			l_execution_type = EXECUTION_TYPE_TRAIN;
			l_i += 2;
		}
		else if(strcmp("--generate", argv[l_i]) == 0)
		{
/* ... */
			pl_model_file_name = (char *)argv[l_i + 1];
			pl_image_file_name = (char *)argv[l_i + 2];
			l_execution_type = EXECUTION_TYPE_GENERATE;
			l_i += 2;
		}
	}

/* ... */
	feedforward_t l_feedforward;
	feedforward_create(&l_feedforward);

/* ... */
	uintmax_t l_hidden_size_buffer[3] = {256, 512, 256};
	feedforward_resize(&l_feedforward, (128 * 128 * 3), l_hidden_size_buffer, 3, (128 * 128 * 3));

/* ... */
	switch(l_execution_type)
	{
/* ... */
	case EXECUTION_TYPE_GENERATE:
	{
/* ... */
		FILE *pl_model_file_handle = fopen(pl_model_file_name, "rb");
		if(!pl_model_file_handle)
		{
			fprintf(stderr, "error: Failed to open the model!\n");
			return -1;
		}

/* ... */
		for(uintmax_t l_i = 0; l_i < l_feedforward.t_hidden_layer_weight_buffer_size; l_i++)
		{
			uintmax_t l_weight_count = (l_i == 0) ? l_hidden_size_buffer[0] * (128 * 128 * 3) : l_hidden_size_buffer[l_i] * l_hidden_size_buffer[l_i-1];
			fread(l_feedforward.ppt_hidden_layer_weight_buffer[l_i], sizeof(float_t), l_weight_count, pl_model_file_handle);
		}

/* ... */
		for(uintmax_t l_i = 0; l_i < l_feedforward.t_hidden_layer_weight_bias_size; l_i++)
		{
			uintmax_t l_bias_count = l_hidden_size_buffer[l_i];
			fread(l_feedforward.ppt_hidden_layer_bias_buffer[l_i], sizeof(float_t), l_bias_count, pl_model_file_handle);
		}

/* ... */
		uintmax_t l_last_hidden_layer_size = (l_feedforward.t_hidden_layer_size_buffer_size > 0) ? l_feedforward.pt_hidden_layer_size_buffer[l_feedforward.t_hidden_layer_size_buffer_size - 1] : (128 * 128 * 3);
		
/* ... */
		fread(l_feedforward.pt_output_layer_weight_buffer, sizeof(float_t), (128 * 128 * 3) * l_last_hidden_layer_size, pl_model_file_handle);
		fread(l_feedforward.pt_output_layer_bias_buffer, sizeof(float_t), (128 * 128 * 3), pl_model_file_handle);
		
/* ... */
		fclose(pl_model_file_handle);

/* ... */
		uintmax_t l_latent_buffer_size = l_hidden_size_buffer[0];  // 512
		float_t *pl_latent_buffer = malloc(l_latent_buffer_size * sizeof(float_t));
		for(uintmax_t l_i = 0; l_i < l_latent_buffer_size; l_i++) pl_latent_buffer[l_i] = (float_t)rand() / (float_t)RAND_MAX;

/* ... */
		uintmax_t l_output_buffer_size = (128 * 128 * 3);
		float_t *pl_output_buffer = malloc(l_output_buffer_size * sizeof(float_t));
		for(uintmax_t l_i = 0; l_i < l_output_buffer_size; l_i++)
		{
/* ... */
			float_t l_sum = 0.0f;
			for(uintmax_t l_j = 0; l_j < l_latent_buffer_size; l_j++) l_sum += l_feedforward.pt_output_layer_weight_buffer[l_j * l_output_buffer_size + l_i] * pl_latent_buffer[l_j];

/* ... */
			l_sum += l_feedforward.pt_output_layer_bias_buffer[l_i];

/* ... */
			if(l_sum < 0.0f) l_sum = 0.0f;
			if(l_sum > 1.0f) l_sum = 1.0f;

/* ... */
			pl_output_buffer[l_i] = l_sum;
		}

/* ... */
		free(pl_latent_buffer);

/* ... */
		FILE *pl_image_file_handle = fopen(pl_image_file_name, "wb");
		if(!pl_image_file_handle)
		{
			fprintf(stderr, "error: Could not open image file!\n");
			return -1;
		}

/* ... */
		uint8_t l_bmp_header[54] = {0};
		uint32_t l_file_size = 54 + (128 * 128 * 3);

/* ... */
		*(uint16_t *)&l_bmp_header[0] = 0x4D42;
		*(uint32_t *)&l_bmp_header[2] = l_file_size;
		*(uint32_t *)&l_bmp_header[10] = 54;
		*(uint32_t *)&l_bmp_header[14] = 40;
		*(uint32_t *)&l_bmp_header[18] = 128;
		*(uint32_t *)&l_bmp_header[22] = 128;
		*(uint16_t *)&l_bmp_header[26] = 1;
		*(uint16_t *)&l_bmp_header[28] = 24;
		*(uint32_t *)&l_bmp_header[34] = (128 * 128 * 3);

/* ... */
		fwrite(l_bmp_header, sizeof(uint8_t), 54, pl_image_file_handle);
		for(uintmax_t l_i = 0; l_i < l_output_buffer_size; l_i++)
		{
			uint8_t l_pixel = (uint8_t)(fmin(fmax(pl_output_buffer[l_i], 0.0f), 1.0f) * 255);
			fwrite(&l_pixel, sizeof(uint8_t), 1, pl_image_file_handle);
		}

/* ... */
		fclose(pl_image_file_handle);

/* ... */
		free(pl_output_buffer);

/* ... */
		break;
	}
/* ... */
	case EXECUTION_TYPE_TRAIN:
	{
/* ... */
		FILE *pl_model_file_handle = fopen(pl_model_file_name, "rb");
		if(pl_model_file_handle)
		{
/* ... */
			for(uintmax_t l_i = 0; l_i < l_feedforward.t_hidden_layer_weight_buffer_size; l_i++)
			{
				uintmax_t l_weight_count = (l_i == 0) ? l_hidden_size_buffer[0] * (128 * 128 * 3) : l_hidden_size_buffer[l_i] * l_hidden_size_buffer[l_i-1];
				fread(l_feedforward.ppt_hidden_layer_weight_buffer[l_i], sizeof(float_t), l_weight_count, pl_model_file_handle);
			}

/* ... */
			for(uintmax_t l_i = 0; l_i < l_feedforward.t_hidden_layer_weight_bias_size; l_i++)
			{
				uintmax_t l_bias_count = l_hidden_size_buffer[l_i];
				fread(l_feedforward.ppt_hidden_layer_bias_buffer[l_i], sizeof(float_t), l_bias_count, pl_model_file_handle);
			}

/* ... */
			uintmax_t l_last_hidden_layer_size = (l_feedforward.t_hidden_layer_size_buffer_size > 0) ? l_hidden_size_buffer[l_feedforward.t_hidden_layer_size_buffer_size - 1] : (128 * 128 * 3);
			fread(l_feedforward.pt_output_layer_weight_buffer, sizeof(float_t), (128 * 128 * 3) * l_last_hidden_layer_size, pl_model_file_handle);
			fread(l_feedforward.pt_output_layer_bias_buffer, sizeof(float_t), (128 * 128 * 3), pl_model_file_handle);

/* ... */
			fclose(pl_model_file_handle);
		}
/* ... */
		else feedforward_random(&l_feedforward);

/* ... */
		FILE *pl_image_file_handle = fopen(pl_image_file_name, "rb");
		if(!pl_image_file_handle)
		{
			fprintf(stderr, "error: Could not open image file!\n");
			return -1;
		}

/* ... */
		fseek(pl_image_file_handle, 54, SEEK_SET);
		uint8_t *pl_pixel_data_buffer = malloc((128 * 128 * 3) * sizeof(uint8_t));
		fread(pl_pixel_data_buffer, sizeof(uint8_t), (128 * 128 * 3), pl_image_file_handle);
		fclose(pl_image_file_handle);

/* ... */
		float_t *pl_input_buffer = malloc((128 * 128 * 3) * sizeof(float_t));
		for(uintmax_t l_i = 0; l_i < (128 * 128 * 3); l_i++) pl_input_buffer[l_i] = pl_pixel_data_buffer[l_i] / 255.0f;
		free(pl_pixel_data_buffer);

/* ... */
		float_t *pl_output_buffer = malloc((128 * 128 * 3) * sizeof(float_t));
		const uintmax_t l_iteration_count = 100;
		const float_t l_learning_rate = 0.01f;
		for(uintmax_t l_i = 0; l_i < l_iteration_count; l_i++)
		{
			feedforward_forward(&l_feedforward, pl_input_buffer, (128 * 128 * 3), pl_output_buffer, (128 * 128 * 3));
/* ... */
			feedforward_backward(&l_feedforward, pl_input_buffer, (128 * 128 * 3), pl_input_buffer, (128 * 128 * 3), l_learning_rate);
			fprintf(stdout, "debug: Iteration %ju/%ju complete.\n", l_i, l_iteration_count);
		}

/* ... */
		free(pl_output_buffer);

/* ... */
		pl_model_file_handle = fopen(pl_model_file_name, "wb");
		if(!pl_model_file_handle)
		{
			fprintf(stderr, "error: Could not open model file!\n");
			return -1;
		}

/* ... */
		for(uintmax_t l_i = 0; l_i < l_feedforward.t_hidden_layer_weight_buffer_size; l_i++)
		{
			uintmax_t l_weight_count = (l_i == 0) ? l_hidden_size_buffer[0] * (128 * 128 * 3) : l_hidden_size_buffer[l_i] * l_hidden_size_buffer[l_i-1];
			fwrite(l_feedforward.ppt_hidden_layer_weight_buffer[l_i], sizeof(float_t), l_weight_count, pl_model_file_handle);
		}

/* ... */
		for(uintmax_t l_i = 0; l_i < l_feedforward.t_hidden_layer_weight_bias_size; l_i++)
		{
			uintmax_t l_bias_count = l_hidden_size_buffer[l_i];
			fwrite(l_feedforward.ppt_hidden_layer_bias_buffer[l_i], sizeof(float_t), l_bias_count, pl_model_file_handle);
		}

/* ... */
		uintmax_t l_last_hidden_layer_size = (l_feedforward.t_hidden_layer_size_buffer_size > 0) ? l_hidden_size_buffer[l_feedforward.t_hidden_layer_size_buffer_size - 1] : (128 * 128 * 3);
		fwrite(l_feedforward.pt_output_layer_weight_buffer, sizeof(float_t), (128 * 128 * 3) * l_last_hidden_layer_size, pl_model_file_handle);
		fwrite(l_feedforward.pt_output_layer_bias_buffer, sizeof(float_t), (128 * 128 * 3), pl_model_file_handle);
		
/* ... */
		fclose(pl_model_file_handle);
		
/* ... */
		free(pl_input_buffer);

/* ... */
		break;
	}
	default:
		break;
	}

/* ... */
	feedforward_delete(&l_feedforward);

/* ... */
	return 0;
}
