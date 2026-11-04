/*
 * Dump LTE reference vectors from srsRAN 4G's PHY library for cross-checking
 * pr450's LTE generator (tests/test_against_srsran.py).
 *
 * For each cell ID and CRS port count, one 10 ms frame (140 symbols x 12*N_RB
 * subcarriers, 5 MHz = 25 PRB, normal CP, FDD) is built with srsRAN's own
 * generators and placement routines:
 *   CRS      srsran_refsignal_cs_put_sf   (every subframe, every port)
 *   PSS/SSS  srsran_pss_put_slot / srsran_sss_put_slot (slot 0 of subframes 0 and 5)
 *
 * Output: <outdir>/lte_c<cell>_p<ports>.bin, a sequence of little-endian records
 *   int32 kind (1 = CRS, 2 = PSS/SSS), int32 symbol, int32 subcarrier, float32 re, float32 im
 * containing every non-zero RE of the frame.
 */
#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "srsran/phy/ch_estimation/refsignal_dl.h"
#include "srsran/phy/common/phy_common.h"
#include "srsran/phy/sync/pss.h"
#include "srsran/phy/sync/sss.h"

#define NOF_PRB 25
#define NSC (NOF_PRB * SRSRAN_NRE)
#define NSYMB_SF 14

static int dump(FILE* f, int kind, const cf_t* grid, int nsym)
{
  int n = 0;
  for (int l = 0; l < nsym; l++) {
    for (int k = 0; k < NSC; k++) {
      cf_t v = grid[l * NSC + k];
      if (crealf(v) != 0.0f || cimagf(v) != 0.0f) {
        int32_t hdr[3] = {kind, l, k};
        float   val[2] = {crealf(v), cimagf(v)};
        fwrite(hdr, sizeof(int32_t), 3, f);
        fwrite(val, sizeof(float), 2, f);
        n++;
      }
    }
  }
  return n;
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    fprintf(stderr, "usage: %s <outdir> <cell_id> [<cell_id> ...]\n", argv[0]);
    return 1;
  }
  const char* outdir = argv[1];
  cf_t*       crs    = calloc(10 * NSYMB_SF * NSC, sizeof(cf_t));
  cf_t*       sync   = calloc(10 * NSYMB_SF * NSC, sizeof(cf_t));

  for (int a = 2; a < argc; a++) {
    uint32_t cell_id = (uint32_t)atoi(argv[a]);
    for (uint32_t ports = 1; ports <= 2; ports++) {
      memset(crs, 0, 10 * NSYMB_SF * NSC * sizeof(cf_t));
      memset(sync, 0, 10 * NSYMB_SF * NSC * sizeof(cf_t));

      srsran_cell_t cell = {0};
      cell.nof_prb       = NOF_PRB;
      cell.nof_ports     = ports;
      cell.id            = cell_id;
      cell.cp            = SRSRAN_CP_NORM;
      cell.frame_type    = SRSRAN_FDD;

      srsran_refsignal_t rs;
      if (srsran_refsignal_cs_init(&rs, NOF_PRB) || srsran_refsignal_cs_set_cell(&rs, cell)) {
        fprintf(stderr, "refsignal init failed\n");
        return 1;
      }

      cf_t  pss[SRSRAN_PSS_LEN];
      float sss0[SRSRAN_SSS_LEN], sss5[SRSRAN_SSS_LEN];
      srsran_pss_generate(pss, cell_id % 3);
      srsran_sss_generate(sss0, sss5, cell_id);

      for (uint32_t sf = 0; sf < 10; sf++) {
        srsran_dl_sf_cfg_t sf_cfg = {0};
        sf_cfg.tti                = sf;
        sf_cfg.cfi                = 2;
        sf_cfg.sf_type            = SRSRAN_SF_NORM;
        cf_t* sf_crs              = &crs[sf * NSYMB_SF * NSC];
        for (uint32_t p = 0; p < ports; p++) {
          srsran_refsignal_cs_put_sf(&rs, &sf_cfg, p, sf_crs);
        }
        if (sf == 0 || sf == 5) {
          cf_t* slot0 = &sync[sf * NSYMB_SF * NSC];
          srsran_pss_put_slot(pss, slot0, NOF_PRB, SRSRAN_CP_NORM);
          srsran_sss_put_slot(sf == 0 ? sss0 : sss5, slot0, NOF_PRB, SRSRAN_CP_NORM);
        }
      }
      srsran_refsignal_free(&rs);

      char path[1024];
      snprintf(path, sizeof(path), "%s/lte_c%u_p%u.bin", outdir, cell_id, ports);
      FILE* f = fopen(path, "wb");
      if (!f) {
        perror(path);
        return 1;
      }
      int n_crs  = dump(f, 1, crs, 10 * NSYMB_SF);
      int n_sync = dump(f, 2, sync, 10 * NSYMB_SF);
      fclose(f);
      printf("cell %3u ports %u: %d CRS + %d PSS/SSS REs -> %s\n", cell_id, ports, n_crs, n_sync, path);
    }
  }
  free(crs);
  free(sync);
  return 0;
}
