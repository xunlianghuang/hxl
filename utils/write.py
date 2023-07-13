import xlsxwriter
import os
def write_to_excel(training_losses, testing_losses, training_accuracies, testing_accuracies,save_path):
    workbook = xlsxwriter.Workbook(os.path.join(save_path,'training_results.xlsx'))
    worksheet = workbook.add_worksheet()

    # Write headers
    worksheet.write(0, 0, 'Training Loss')
    worksheet.write(0, 1, 'Testing Loss')
    worksheet.write(0, 2, 'Training Accuracy')
    worksheet.write(0, 3, 'Testing Accuracy')

    # Write data
    for i in range(len(training_losses)):
        # print(i) #0 1 2 3
        worksheet.write(i+1, 0, training_losses[i])
        worksheet.write(i+1, 1, testing_losses[i])
        worksheet.write(i+1, 2, training_accuracies[i])
        worksheet.write(i+1, 3, testing_accuracies[i])

    workbook.close()