using FaultDetection.Web.Data;
using FaultDetection.Web.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace FaultDetection.Web.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class FaultsController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly IWebHostEnvironment _environment;

        public FaultsController(ApplicationDbContext context, IWebHostEnvironment environment)
        {
            _context = context;
            _environment = environment;
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<InspectionLog>>> GetRecentFaults()
        {
            return await _context.InspectionLogs
                .OrderByDescending(x => x.CreatedDate)
                .Take(50)
                .ToListAsync();
        }

        [HttpPost]
        public async Task<IActionResult> ReportFault([FromBody] InspectionLog log)
        {
            if (log == null) return BadRequest();

            log.CreatedDate = DateTime.UtcNow;
            _context.InspectionLogs.Add(log);
            await _context.SaveChangesAsync();

            return Ok(new { message = "Fault reported successfully", id = log.Id });
        }

        [HttpPost("upload-image")]
        public async Task<IActionResult> UploadImage(IFormFile image)
        {
            if (image == null || image.Length == 0)
                return BadRequest("No image uploaded.");

            var uploadsPath = Path.Combine(_environment.WebRootPath, "uploads");
            if (!Directory.Exists(uploadsPath))
                Directory.CreateDirectory(uploadsPath);

            var fileName = $"{Guid.NewGuid()}{Path.GetExtension(image.FileName)}";
            var filePath = Path.Combine(uploadsPath, fileName);

            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await image.CopyToAsync(stream);
            }

            return Ok(new { imagePath = $"/uploads/{fileName}" });
        }
    }
}
